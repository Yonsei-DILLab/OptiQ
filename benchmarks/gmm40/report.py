"""Render and archive analysis from an already completed common evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb

from optiq_dime.runtime import load_environment, ROOT
from .train import write_json


def build_report(directory):
    summary = json.loads((directory/'summary.json').read_text())
    rows = {r['method']: r for r in summary['records']}
    mean, sd = summary['optiq_three_seed_mean'], summary['optiq_three_seed_sd']
    config = summary['sources']['OptiQ seed 0']['config']
    n_rows = config['num_policy_samples']
    n_candidates = n_rows * config['proposals_per_policy_sample']
    sigma = config['proposal_std'] * config['coordinate_scale']
    include_anchor = config.get('include_anchor', False)
    unbounded = config.get('unbounded_actions', False)
    coordinate_scale = config['coordinate_scale']
    coordinate_note = (
        f'GMM 원래 좌표에서 x=g(z)를 직접 사용했다. 입력·출력 좌표 변환 없이 KDE sigma={sigma:g}를 적용했다.'
        if coordinate_scale == 1. and unbounded else
        f"좌표는 {'u∈R²' if unbounded else 'u∈[-1,1]²'}에서 x={coordinate_scale:g}u로 변환했다. KDE sigma={config['proposal_std']}는 원래 좌표 sigma={sigma:g}에 해당한다."
    )
    boundary_note = (
        'GMM 전용 unbounded 경로로 후보 perturbation 제한·action 경계·생성기 출력 clipping을 제거했다. '
        'Gaussian KDE의 샘플링과 log density 모두 같은 무제한 정규분포를 사용한다.'
        if unbounded else
        '기존 action 경계와 perturbation 제한을 사용하는 truncated Gaussian KDE 경로다.'
    )
    anchor_text = '포함' if include_anchor else '없음'
    weight_interpretation = (
        '기존 코드의 현재 가중치는 anchor와 무작위 후보 모두에서 `w ∝ p(x)^4 / q_KDE(x)^0.1`이다. '
        'Anchor는 KDE 중심점의 결정적 복사이며, 기존 코드대로 연속 KDE 밀도를 적용했다. '
        '따라서 전체 후보가 q_KDE에서 무작위로 추출되었다고 볼 수 없고, '
        '`p^4 q_KDE^0.9`를 전체 가중 후보 분포라고 해석해서는 안 된다. '
        '이 실험은 기존 anchor 경로의 성능을 측정하며, 혼합 측도의 정확한 중요도 보정을 새로 구현하지 않았다.'
        if include_anchor else
        '기존 코드의 현재 가중치는 `w ∝ p(x)^4 / q_KDE(x)^0.1`이다. 고정 q에서 이상적인 가중 후보 분포는 `p(x)^4 q_KDE(x)^0.9`에 비례한다. 여기에 유한 후보, Sinkhorn, row argmax, 신경망 증류가 추가된다. 따라서 원본 p를 복원하도록 설계된 완전한 중요도 보정과 같지 않다.'
    )
    budget = summary['sources']['OptiQ seed 0']['completion']['final_metrics']['target_density_evaluations']
    counts = [rows[f'OptiQ seed {seed}']['modes_covered'] for seed in (0,1,2)]
    counts_text = ' / '.join(str(n) for n in counts)
    diagnostics = []
    all_eval_means = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fields = [('eval/mean_log_prob', 'Mean log p(x)'), ('eval/modes_covered', 'Modes covered / 40'),
        ('train/source_ess_absolute', f'Source ESS / {n_candidates}'), ('eval/mode_occupancy_tvd', 'Mode occupancy TVD')]
    for seed in (0, 1, 2):
        source = summary['sources']['OptiQ seed '+str(seed)]
        path = Path(source['path']).parent
        records = [json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
        by_step = {r['update']: r for r in records if 'eval/mean_log_prob' in r}
        all_eval_means.extend(r['eval/mean_log_prob'] for r in by_step.values())
        late = [r for r in records if 'train/source_ess_absolute' in r and r['update'] >= 25000]
        diagnostics.append({'seed': seed, 'wandb_url': source['config']['wandb_url'],
            'training_seconds': source['completion']['elapsed_seconds'],
            'late_ess': float(np.mean([r['train/source_ess_absolute'] for r in late])),
            'late_max_weight': float(np.mean([r['train/max_source_weight'] for r in late])),
            'late_q_std': float(np.mean([r['train/source_q_std'] for r in late])),
            'best_logged_mode_count': max(r['eval/modes_covered'] for r in by_step.values()),
            'final_mode_count': by_step[30000]['eval/modes_covered']})
        for field in ['local_anchor_argmax_fraction', 'local_best_q_gain_over_anchor', 'local_improvement_fraction']:
            values = [r['train/'+field] for r in late if 'train/'+field in r]
            if values:
                diagnostics[-1]['late_'+field] = float(np.mean(values))
        for ax, (field, label) in zip(axes.flat, fields):
            values = list(by_step.values()) if field.startswith('eval/') else [r for r in records if field in r]
            ax.plot([r['update'] for r in values], [r[field] for r in values], label='seed '+str(seed), lw=1.4, alpha=.85)
            ax.set_xlabel('Actor updates'); ax.set_ylabel(label); ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
    ref_mean = rows['DiKL']['reference_mean_log_prob']
    axes[0,0].axhline(ref_mean, color='black', ls='--', lw=1, label='GT reference')
    logp_upper = max(-5.5, float(np.ceil((max(all_eval_means) + .1)*10)/10))
    axes[0,0].set_ylim(-10, logp_upper)  # Initial values outside this range are documented.
    axes[0,0].legend(fontsize=8)
    axes[0,1].set_ylim(0,41)
    axes[1,0].set_ylim(0,n_candidates)
    axes[1,1].set_ylim(0,1)
    fig.suptitle(f'GMM40: {n_rows}x{n_candidates} OT, sigma={sigma:g}, anchor={include_anchor}, unbounded={unbounded}\nT=0.25 fixed, beta=0.1, coordinate scale={coordinate_scale:g}')
    fig.text(.07,.015,f'Evaluation: 10,000 direct generator samples, fixed independent latent keys.\nLog-density panel zooms to [-10, {logp_upper:g}]; lower initial values are outside the displayed range.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.96))
    for ext in ['png','pdf']:
        fig.savefig(directory/('training.'+ext),dpi=170)
    plt.close(fig)
    lines = ['# OptiQ g(z) GMM40 결과: T=0.25 고정', '',
        f'최종 모드 커버리지는 seed 0/1/2에서 **{counts_text} / 40**이다. log p가 높아지는 현상만으로 분포 복원 성능 향상이라고 읽으면 안 된다.', '',
        '## 실행과 측정', '',
        '- heechan-no-anchor의 기존 `ImplicitActor`와 `OptiQDIME.update_actor`를 직접 재사용했다. 상태 입력을 비우고 Q만 정확한 GMM log p로 바꾸었다. 학습한 critic이나 MuJoCo 체크포인트는 사용하지 않았다.',
        f"- T=0.25 고정, density beta=0.1, anchor {anchor_text}, {n_rows}×{n_candidates} OT, 배치 {config['batch_size']}, 256×3 GELU, Adam lr=0.0003. 기본 actor의 초기화·가중치 계산·Sinkhorn·argmax·MSE 경로를 유지했다.",
        f"- KDE 중심 {n_rows}개; 중심마다 anchor {int(include_anchor)}개 + 무작위 후보 {config['proposals_per_policy_sample'] - int(include_anchor)}개. Candidate density는 추출 전 구성한 같은 KDE로 계산했다.",
        '- '+boundary_note,
        f'- Seed 0/1/2 각각 30,000회 업데이트 완료. 훈련 target-density 평가 {budget:,}개/seed. 최종 체크포인트에서 나온 직접 g(z) 출력 10,000개씩 평가. KDE 후보를 결과 샘플로 쓰지 않았다.',
        '- '+coordinate_note+' GMM 성분 표준편차는 softplus(1)=1.31326이다.',
        f'- 공통 CPU reference seed=20260821, 10,000개 정답 샘플의 평균 log p는 **{ref_mean:.6f}**다. 논문의 별도 정답 표 값은 -6.85이다.', '',
        '## 동일 평가기로 계산한 비교', '',
        'OptiQ의 ±는 3개 학습 seed 간 표준편차다. 공개 baseline은 제공된 샘플 파일의 한 집합이므로 seed 간 편차를 붙이지 않는다.', '',
        '| 방법 | mean log p | GT 평균과 절대 오차 ↓ | 모드 /40 | Occupancy TVD ↓ | Sliced W2 ↓ | Sample W2 ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['DiKL','iDEM','FAB','R-KL']:
        r=rows[name]
        lines.append(f"| {name} 공개 샘플 | {r['mean_log_prob']:.4f} | {r['mean_log_prob_abs_error']:.4f} | {r['modes_covered']} | {r['mode_occupancy_tvd']:.4f} | {r['sliced_w2']:.4f} | {r['sample_w2']:.4f} |")
    lines.append(f"| **OptiQ 3 seeds** | **{mean['mean_log_prob']:.4f} ± {sd['mean_log_prob']:.4f}** | **{mean['mean_log_prob_abs_error']:.4f} ± {sd['mean_log_prob_abs_error']:.4f}** | **{counts_text}** | **{mean['mode_occupancy_tvd']:.4f} ± {sd['mode_occupancy_tvd']:.4f}** | **{mean['sliced_w2']:.4f} ± {sd['sliced_w2']:.4f}** | **{mean['sample_w2']:.4f} ± {sd['sample_w2']:.4f}** |")
    lines += ['', 'Sample W2는 각 방법에서 2,000개씩 10회 재추출해 정확한 empirical OT를 계산한 평균이다. 기준 target-vs-target의 유한 표본 바닥은 W2 약 3.251, Sliced W2 약 0.559이다. 세부 재추출 편차는 CSV/JSON에 따로 저장했다.', '',
        '## 논문 값과의 관계', '',
        '[DiKL 논문 Table 1](https://arxiv.org/html/2410.12456v2#S5.T1)은 GMM40에서 mean log p를 True -6.85, FAB -10.74, iDEM -8.33, DiKL -7.21로 보고한다. 공개 DiKL·iDEM 샘플은 해당 값을 재현했다. FAB는 전체 공개 파일의 log p와 공통 10,000개 부분집합의 값을 `summary.json`에 모두 보존했다.',
        '', 'Coverage·W2·TVD 등 이 보고서의 추가 GMM 수치는 공개 샘플을 같은 평가기로 재측정한 결과다. 이를 논문에 실린 GMM 표 값이라고 표현하지 않는다. 논문은 다른 네트워크와 2.5시간 훈련 예산을 사용하므로 같은 예산의 알고리즘 비교도 아니다.', '',
        '## 왜 log p만 보면 오판하는가', '',
        f"OptiQ 평균 log p는 {mean['mean_log_prob']:.3f}, GT는 {ref_mean:.3f}다. 밀도 평균이 높더라도 정답 분포처럼 모든 모드와 그 주변을 채운다는 뜻은 아니다. 실제 coverage는 {counts_text}, occupancy TVD는 약 {mean['mode_occupancy_tvd']:.3f}이다. 이 지표들을 함께 해석해야 한다.",
        '', 'R-KL도 평균 log p 절대 오차만 보면 DiKL보다 작지만 8개 모드만 덮는다. 따라서 log p 평균뿐 아니라 절대 오차와 coverage·transport를 함께 봐야 한다.', '',
        '## 가중치와 진단', '',
        weight_interpretation, '',
        f'| seed | 마지막 5k 업데이트 ESS /{n_candidates} | 최대 가중치 평균 | 최종 모드 | 기록상 최대 모드 |', '|---|---:|---:|---:|---:|']
    for r in diagnostics:
        lines.append(f"| {r['seed']} | {r['late_ess']:.3f} | {r['late_max_weight']:.3f} | {r['final_mode_count']} | {r['best_logged_mode_count']} |")
    if include_anchor:
        lines += ['', '아래 진단은 각 KDE 중심의 후보 그룹 안에서 계산한 값이다. Anchor argmax 비율은 그룹 내 Q 최대 후보가 anchor인 비율이며, OT 이후 anchor가 선택된 비율과 다르다. 마지막 5k 업데이트 평균이다.', '',
            '| seed | 그룹 내 anchor argmax 비율 | 무작위 후보가 anchor보다 Q가 높은 비율 | 그룹 내 최고 Q − anchor Q |', '|---|---:|---:|---:|']
        for r in diagnostics:
            if 'late_local_anchor_argmax_fraction' in r:
                lines.append(f"| {r['seed']} | {r['late_local_anchor_argmax_fraction']:.4f} | {r['late_local_improvement_fraction']:.4f} | {r['late_local_best_q_gain_over_anchor']:.4f} |")
    lines += ['', '이 실험에서는 Q가 정확하므로 coverage 부족을 learned critic 추정 오차로 설명할 수 없다. 다만 T, beta, KDE 폭, argmax 증류 각각의 기여는 분리하지 않았으므로 특정 요소 하나가 원인이라고 확정할 수 없다.',
        '', 'T=0.25와 beta=0.1을 유지한 결과만 측정했다. T=1/beta=1 대조군이나 gradient 변형 실험은 실행하지 않았다.', '',
        '## 재현 파일과 W&B', '',
        '- [실험 설정 및 코드 이식 설명](../../../docs/GMM40_BENCHMARK.md)',
        '- [최종 샘플 그림](samples.png) · [PDF](samples.pdf) · [학습 추이](training.png)',
        '- [지표 CSV](comparison.csv) · [전체 수치·출처](summary.json) · [OT 재추출 기록](transport_repeats.csv)',
        f"- [공통 평가 W&B]({summary['wandb_url']})"]
    for r in diagnostics:
        lines.append(f"- [OptiQ seed {r['seed']}]({r['wandb_url']}) — 30k 학습/평가 {r['training_seconds']:.1f}초 (W&B 최종 업로드 시간 제외)")
    (directory/'analysis.md').write_text('\n'.join(lines)+'\n')
    metadata = {'training_diagnostics': diagnostics, 'evaluation_source_sha256': {}}
    for relative in ['benchmarks/gmm40/evaluate.py','benchmarks/gmm40/metrics.py','benchmarks/gmm40/target_torch.py','benchmarks/gmm40/report.py']:
        data = (ROOT/relative).read_bytes()
        metadata['evaluation_source_sha256'][relative] = hashlib.sha256(data).hexdigest()
        path = directory/'evaluation_source'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_json(directory/'analysis_metadata.json',metadata)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    summary = build_report(directory)
    load_environment()
    run = wandb.init(id=summary['wandb_url'].rsplit('/',1)[1], resume='must',
        project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'),mode='online',dir=str(directory))
    try:
        run.log({'training_curves': wandb.Image(str(directory/'training.png'))})
        artifact = wandb.Artifact('gmm40-final-analysis-'+run.id, type='analysis')
        for path in directory.iterdir():
            if path.is_file():
                artifact.add_file(str(path),name=path.name)
        artifact.add_dir(str(directory/'evaluation_source'),name='evaluation_source')
        run.log_artifact(artifact)
        run.summary['analysis_completed'] = True
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    print(directory/'analysis.md')


if __name__ == '__main__':
    main()
