"""Postprocess saved SAME-cloud assignments; no training, resampling, or OT solve."""
import argparse
import base64
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

METHODS = ['sinkhorn', 'exact', 'sinkhorn_same_quantized_teacher']
LABELS = {'sinkhorn': 'Sinkhorn row-argmax', 'exact': 'Exact OT row-argmax',
          'monge': 'Quantile Monge',
          'sinkhorn_same_quantized_teacher': 'Sinkhorn argmax (quantized teacher)'}
COLORS = {'sinkhorn': '#ce602e', 'exact': '#148391', 'monge': '#7650ad'}
CASES = ['double_mass', 'tri_split']
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def cdf_error(b, mass, ref):
    order = np.argsort(b, kind='stable')
    return float(np.max(np.abs(np.cumsum((mass-ref)[order]))))


def analyze(path, task, boundaries):
    with np.load(path) as d:
        x, b, w = [np.asarray(d[k], dtype=float).ravel() for k in ('x', 'b', 'w')]
        w /= w.sum()
        n = len(x)
        mi = d['monge_plan'].argmax(1)
        monge = b[mi]
        order = np.argsort(x, kind='stable')
        assert np.all(np.diff(monge[order]) >= -1e-7)
        assert np.allclose(d['monge_plan'].sum(1), 1/n)
        rows = []
        for method in METHODS:
            p = np.asarray(d[method+'_plan'], dtype=float)
            indices = p.argmax(1)
            assert np.array_equal(indices, d[method+'_effective'].argmax(1))
            assert np.isfinite(p).all() and (p.sum(1) > 0).all()
            selected = b[indices]
            delta = np.abs(selected-monge)
            mass = np.bincount(indices, minlength=len(b))/n
            # Uniformly mix conditional rows; this separates finite row residuals
            # from the additional effect of replacing each row with its argmax.
            soft_mass = (p/p.sum(1, keepdims=True)).mean(0)
            teacher = w if method != 'sinkhorn_same_quantized_teacher' else np.asarray(d['quantized_w'], float)
            teacher /= teacher.sum()
            rows.append(dict(
                run=task['name'], case=task['case'], origin=task['method'], seed=task['seed'],
                step=int(path.stem), comparison=method, n=n, m=len(b),
                candidate_match=float(np.mean(indices == mi)), action_mae=float(delta.mean()),
                action_median=float(np.median(delta)), action_p90=float(np.quantile(delta, .9)),
                within_005=float(np.mean(delta <= .05)),
                same_mode=float(np.mean(np.searchsorted(boundaries[1:-1], selected, side='right') ==
                                        np.searchsorted(boundaries[1:-1], monge, side='right'))),
                selected_unique=len(np.unique(indices)), monge_unique=len(np.unique(mi)),
                max_row_share=float(mass.max()),
                soft_teacher_cdf=cdf_error(b, soft_mass, teacher),
                hard_teacher_cdf=cdf_error(b, mass, teacher),
                monge_original_teacher_cdf=cdf_error(b, np.bincount(mi, minlength=len(b))/n, w),
                row_mass_l1=float(np.abs(p.sum(1)-1/n).sum()),
                column_mass_l1=float(np.abs(p.sum(0)-teacher).sum()),
                zero_rows=int(np.sum(p.sum(1) == 0)),
                exact_max_tie_fraction=float(np.mean((p == p.max(1, keepdims=True)).sum(1) > 1)),
                source_tie_fraction=float(np.mean(np.diff(x[order]) == 0)),
            ))
        return rows


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def render(md, output):
    body = markdown.markdown(md, extensions=['tables', 'fenced_code'])
    for p in output.glob('*.png'):
        body = body.replace('src="'+p.name+'"', 'src="data:image/png;base64,'+
                            base64.b64encode(p.read_bytes()).decode()+'"')
    return ('<!doctype html><meta charset="utf-8"><title>Sinkhorn argmax vs Monge</title>'
            '<style>body{max-width:1200px;margin:40px auto;font-family:sans-serif;line-height:1.65;'
            'padding:0 20px}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;'
            'border:1px solid #ddd}code{overflow-wrap:anywhere}</style>'+body)


def figures(root, output, rows, reference):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    # Preselected final step, seed 0, both cases; never select a best seed.
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for r, case in enumerate(CASES):
        path = root/'runs'/f'{case}_argmax_s0'/'assignment_audits'/'035000.npz'
        with np.load(path) as d:
            b = d['b'].ravel(); monge = b[d['monge_plan'].argmax(1)]
            ax = axs[r, 0]
            ax.plot([-1, 1], [-1, 1], '--', color='#9a9a9a', lw=1, label='Identical action')
            for method in ['sinkhorn', 'exact']:
                a = b[d[method+'_plan'].argmax(1)]
                ax.scatter(monge, a, s=12 if method == 'sinkhorn' else 8,
                           alpha=.65, color=COLORS[method], label=LABELS[method])
            ax.set(title=f'{case} | same row: selected actions', xlabel='Monge-selected action',
                   ylabel='Row-argmax-selected action', xlim=(-1, 1), ylim=(-1, 1))
            ax.set_aspect('equal', adjustable='box'); ax.legend(fontsize=8)
            ax = axs[r, 1]; edges = np.linspace(-1, 1, 65); widths = np.diff(edges)
            teacher = np.histogram(b, edges, weights=d['w'])[0]/widths
            ax.stairs(teacher, edges, color='#697888', fill=True, alpha=.24, label='Weighted teacher (M=1,024)')
            for method in ['monge', 'sinkhorn']:
                a = b[d[method+'_plan'].argmax(1)]
                hist = np.histogram(a, edges)[0]/len(a)/widths
                ax.stairs(hist, edges, color=COLORS[method], lw=1.5, label=LABELS[method]+' targets (N=256)')
            ax.set(title=f'{case} | distribution of selected targets', xlabel='Action', ylabel='Density', xlim=(-1, 1))
            ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.suptitle('Same source / candidates / weights | baseline cloud | seed 0 | update 35,000\n'
                 'Sinkhorn epsilon 0.05, 30 iterations | target histograms: 64 bins, no smoothing', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(output/'selection_comparison.png', dpi=180); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for r, case in enumerate(CASES):
        for method in ['sinkhorn', 'exact']:
            for seed in [0, 1]:
                rr = sorted([v for v in rows if v['case'] == case and v['origin'] == 'argmax'
                             and v['comparison'] == method and v['seed'] == seed and v['step'] > 1], key=lambda v:v['step'])
                for c, key in enumerate(['action_mae', 'same_mode']):
                    axs[r, c].plot([v['step'] for v in rr], [v[key]*(100 if c else 1) for v in rr],
                                   marker='o', ms=4, color=COLORS[method], ls='-' if seed == 0 else ':',
                                   label=LABELS[method] if seed == 0 else None)
        for c in range(2):
            axs[r, c].set(title=case, xlabel='Actor update (saved audits only)',
                          ylabel=['Mean absolute action gap to Monge', 'Same target basin (%)'][c])
            axs[r, c].grid(alpha=.2)
        axs[r, 0].set_ylim(bottom=0); axs[r, 1].set_ylim(0, 102)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Same-cloud counterfactual on baseline runs | solid: seed 0, dotted: seed 1\n'
                 'Connected lines link sparse audits; they are not per-update measurements', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(output/'selection_tracking.png', dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analysis-commit', required=True); p.add_argument('--integrate', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.base))
    from nsq.problems import schedule, analytic_reference
    tasks = json.loads((args.root/'tasks.json').read_text())
    reference = {}; rows = []; manifest = {}
    for task in tasks:
        for path in sorted((args.root/'runs'/task['name']/'assignment_audits').glob('*.npz')):
            key = (task['case'], int(path.stem))
            if key not in reference:
                family, stage = task['case'].split('_')
                reference[key] = analytic_reference(schedule(stage, key[1], family), 1)
            density_path = path.parent.parent/'density'/path.name
            if density_path.exists():
                with np.load(density_path) as density:
                    assert np.allclose(reference[key]['boundaries'], density['boundaries'])
            rows += analyze(path, task, reference[key]['boundaries'])
            manifest[str(path.relative_to(args.root))] = SHA(path)
    assert len(manifest) == 96 and len(rows) == 288
    with (args.output/'selection_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/'selection_metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
    figures(args.root, args.output, rows, reference)
    final = [r for r in rows if r['origin'] == 'argmax' and r['step'] == 35000]
    lines = ['# Sinkhorn row-argmax는 Monge와 비슷한 action을 고르는가?',
             '**Mode 선택은 대체로 비슷하지만, mode 내부의 action 선택은 상당히 다르다. 현재 설정에서는 exact OT의 row-argmax가 Monge에 훨씬 가깝다.**',
             '새 학습은 실행하지 않았다. 완료된 12 runs의 저장된 assignment audit 96개를 사용했다. 각 audit에서 source 256개, candidate 1,024개와 importance weight를 고정하고 세 선택법을 적용했으므로, 서로 다른 actor가 만든 샘플을 직접 비교하는 오류는 없다.',
             '## 비교 방식과 지표',
             '기본 설정은 legacy implicit actor, temperature 0.25, truncated Gaussian KDE std 0.2 / clip 0.5, mean-normalized squared cost, Sinkhorn epsilon 0.05 / 30 iterations다. Monge는 weighted teacher에서 뽑은 256개 midpoint-quantile 대표점에 대한 정렬 bijection이다. 원래 1,024개 arbitrary-weight teacher에 대한 정확한 Monge map은 아니다.',
             '동일한 row i에 대해 Sinkhorn은 `b[argmax_j P_sinkhorn[i,j]]`, exact OT는 `b[argmax_j P_exact[i,j]]`, Monge는 그 row에 배정된 quantile 대표점을 선택한다. 동일 candidate 비율은 index까지 같은 비율이다. Candidate가 달라도 서로 가까울 수 있으므로 **평균 action 거리**와 **거리 ≤ 0.05 비율**, 그리고 **같은 mode 영역을 고르는 비율**을 함께 본다.',
             'Action 범위는 [-1,1]이다. 두 mode 문제의 Gaussian 폭은 0.12, 세/여섯 mode 문제는 0.1이다. 예를 들어 action 거리가 0.08이면 전체 범위의 4%지만, 개별 mode 폭과 비교하면 작다고 보기 어렵다. Mode 영역은 해당 시점의 정확한 target density에서 peak 사이 valley로 나누었다.',
             '주 표는 **baseline actor가 만든 최종 35K cloud**, seed 0,1 평균이다. 나머지 방법이 만든 cloud와 변화 중 시점도 아래에 별도로 집계했다. 96개 audit는 독립 seed 96개가 아니며, 같은 run의 반복 관측이다.',
             '## 최종 시점의 row별 선택: seed 0,1 평균',
             '| 환경 | Monge와 비교한 방법 | 같은 candidate | 같은 mode | 평균 action 거리 | 거리 ≤ 0.05 | 서로 다른 candidate 수 / 256 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for case in CASES:
        for method in ['sinkhorn', 'exact']:
            rr = [v for v in final if v['case'] == case and v['comparison'] == method]
            lines.append(f'| {case} | {LABELS[method]} | {mean(rr,"candidate_match"):.1%} | {mean(rr,"same_mode"):.1%} | '
                         f'{mean(rr,"action_mae"):.5f} | {mean(rr,"within_005"):.1%} | {mean(rr,"selected_unique"):.1f} |')
    lines += ['이 네 cloud에서 Monge는 각각 256개의 서로 다른 candidate를 선택했다. Sinkhorn은 여러 row가 같은 candidate를 고르며, 같은 mode에서도 더 좁은 action 범위를 선택하는 양상이 나타난다. Exact OT의 candidate index 일치율도 100%는 아니지만, 실제 action 거리는 약 0.0015로 훨씬 작다.',
              '![같은 row의 선택과 target histogram](selection_comparison.png)',
              '**그림 읽는 법:** 왼쪽은 한 점이 동일한 source row 하나다. 가로축은 Monge가 고른 action, 세로축은 각 argmax가 고른 action이다. 회색 대각선과 가까울수록 같은 위치를 고른다. 두 환경 모두 최종 35K, baseline seed 0을 표시했다. 오른쪽은 1,024개 후보의 weighted teacher와 각 방식이 고른 256개 target의 histogram이다. 64 bins, smoothing 없음. **Actor의 32,768개 평가 sample 그림이 아니다.**',
              '## 어떤 mode인가와 mode 안에서 어디인가를 구분해야 한다',
              'Sinkhorn argmax도 대부분 Monge와 같은 mode로 보내므로, 멀리서 보면 비슷한 map처럼 보일 수 있다. 하지만 Monge는 teacher의 누적 질량을 따라 mode의 양쪽 꼬리까지 배정한다. 반면 현재 Sinkhorn argmax는 같은 mode 안에서도 일부 위치에 선택을 모은다. 따라서 **mode coverage가 비슷하더라도 분포 모양은 다를 수 있다.**',
              '이것은 기존 학습 결과에서 baseline의 basin TV는 작지만 histogram TV가 컸던 결과와 일관된다. 다만 이 counterfactual 한 번만으로 누적 학습 실패의 원인을 전부 증명한 것은 아니다.',
              '## Row argmax가 teacher 질량을 얼마나 바꾸는가?',
              '**CDF 최대 오차**는 action x 이하에 누적된 선택 질량과 teacher 질량의 차이를 x 전체에서 최대화한 값이다. 예를 들어 0.18이면 어떤 경계의 한쪽에 배분된 질량이 teacher와 18%p 다르다는 뜻이다. Mode 질량 오차나 histogram TV와는 다르다.',
              '유한 iteration 때문에 원래 Sinkhorn plan의 row 합이 정확히 1/N이 아닐 수 있다. 이를 구분하기 위해 각 row를 합 1로 정규화한 뒤 row들을 균등하게 섞은 **soft row mixture**도 계산했다. 이는 각 row 전체를 사용했을 때의 분포다. 같은 row를 argmax 한 점으로 바꾼 hard target과 비교한다.',
              '| 환경 | Soft row mixture → teacher CDF 오차 | Argmax target → teacher CDF 오차 | Monge target → teacher CDF 오차 |',
              '|---|---:|---:|---:|']
    for case in CASES:
        rr = [r for r in final if r['case'] == case and r['comparison'] == 'sinkhorn']
        lines.append(f'| {case} | {mean(rr,"soft_teacher_cdf"):.4f} | {mean(rr,"hard_teacher_cdf"):.4f} | {mean(rr,"monge_original_teacher_cdf"):.4f} |')
    lines += ['**현재 저장된 plan에서도 row 전체를 쓰는 것과 argmax 한 점을 고르는 것은 큰 차이를 만든다.** 최종 네 cloud의 Sinkhorn 원래 column L1 residual은 2.5e-7 미만이고, row L1 residual은 0.0042–0.0652다. 위 soft-mixture 보정 후에도 hard argmax에서 CDF 오차가 크게 증가한다.',
              '이는 epsilon 효과만을 분리한 실험은 아니다. epsilon 0.05와 30 iterations라는 실제 구현 설정의 결과이며, 더 작은 epsilon이나 완전히 수렴한 Sinkhorn의 결과까지 같다고 일반화하지 않는다. 초기/변화 직후의 residual은 더 클 수 있고 모든 값은 CSV에 보관했다.',
              '## 변화 중 시점과 다른 actor의 cloud에서도 확인',
              '![변화 중 row별 선택 비교](selection_tracking.png)',
              '위 그림은 baseline의 저장된 7개 시점(20K, 20K+1, 22K, 25K+1, 27K, 30K+1, 35K)만 비교한다. 실선 seed 0, 점선 seed 1이다. 점 사이 선은 시각적 연결이며, 모든 update를 측정한 곡선이 아니다.',
              '아래는 동일한 7개 시점 × 2 seeds 평균이다. 비교 입력의 출처를 나누어, baseline에서 얻은 결과만 골라 해석하지 않았다. 공통 random initialization의 step 1은 CSV에 포함하되 이 학습 후 집계에서는 제외했다.',
              '| 환경 | Cloud를 만든 actor | 비교 방법 | 같은 mode | 평균 action 거리 | 같은 candidate |',
              '|---|---|---|---:|---:|---:|']
    for case in CASES:
        for origin in ['argmax', 'exact_argmax', 'monge_quantile']:
            for method in ['sinkhorn', 'exact']:
                rr = [v for v in rows if v['case'] == case and v['origin'] == origin and v['step'] > 1 and v['comparison'] == method]
                lines.append(f'| {case} | {origin} | {LABELS[method]} | {mean(rr,"same_mode"):.1%} | {mean(rr,"action_mae"):.5f} | {mean(rr,"candidate_match"):.1%} |')
    lines += ['## 결론과 원본',
              '**“Sinkhorn row-argmax가 대략 같은 mode를 선택한다”는 해석은 맞는다. “Monge와 비슷하게 teacher 분포를 보존하는 action 배정이다”라는 해석은 현재 결과로는 맞지 않는다. Exact OT row-argmax가 Monge의 선택에 훨씬 가깝다.**',
              'Same-quantized-teacher Sinkhorn 비교도 원본 CSV에 포함했다. 그 조건의 teacher 오차는 quantized teacher에 대해 새로 계산하여 원래 teacher와 혼동하지 않았다.',
              '[전체 audit별 지표](selection_metrics.csv) · [기존 학습 결과 보고서](report.md) · [후처리 프로토콜](SELECTION_PROTOCOL.md)',
              f'학습 commit: `6510585036ab7221df0fc5022a970ef27e068061`. 분석 commit: `{args.analysis_commit}`. 기존 학습 source, checkpoint, audit를 변경하지 않았다. 입력 audit 96개 SHA256과 분석 source SHA는 `selection_provenance.json`에 저장했다.']
    text = '\n\n'.join(lines)+'\n'
    text = re.sub(r'(\|[^\n]*\|)\n\n(?=\|)', r'\1\n', text)
    (args.output/'selection_comparison.md').write_text(text)
    (args.output/'selection_comparison.html').write_text(render(text, args.output))
    if args.integrate:
        path = args.output/'report.md'
        original = path.read_text().split('\n<!-- SELECTION_ANALYSIS -->')[0]
        original = re.sub(r'(\|[^\n]*\|)\n\n(?=\|)', r'\1\n', original)
        appendix = '\n<!-- SELECTION_ANALYSIS -->\n\n## 추가 분석: Sinkhorn argmax와 Monge가 고르는 action\n\n'
        appendix += ('같은 source·candidate·weight를 고정하면, Sinkhorn argmax는 Monge와 **mode는 대체로 같지만 mode 내부 위치는 다르게** 고른다. '
                     '최종 baseline cloud의 두 seed 평균에서 mode 일치율은 96.7–98.8%, 평균 action 거리는 0.068–0.082다. '
                     'Exact OT row-argmax의 평균 거리는 두 환경 모두 약 0.0015로 훨씬 작다.\n\n'
                     '![동일 cloud에서 선택한 action 비교](selection_comparison.png)\n\n'
                     '최종 35K, baseline seed 0. 왼쪽: 동일 row의 Monge 선택 vs argmax 선택. 오른쪽: N=256 선택 target histogram, '
                     '64 bins, smoothing 없음; 회색은 M=1,024 weighted teacher. Actor 평가 sample과 구분한다.\n\n'
                     '[상세 비교 보고서](selection_comparison.md) · [그림 포함 HTML](selection_comparison.html) · '
                     '[96개 audit의 지표](selection_metrics.csv)\n')
        path.write_text(original+appendix)
        (args.output/'report.html').write_text(render(original+appendix, args.output))
    provenance = dict(generated_utc=datetime.now(timezone.utc).isoformat(), analysis_commit=args.analysis_commit,
                      training_commit='6510585036ab7221df0fc5022a970ef27e068061',
                      source_sha256=SHA(Path(__file__)), audits_sha256=manifest,
                      reference_source_sha256={k:SHA(args.base/'nsq'/k) for k in ['problems.py', 'config.py']},
                      audit_count=len(manifest), rows=len(rows), new_training=False)
    (args.output/'selection_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'audits':len(manifest), 'rows':len(rows), 'final_baseline':final}, indent=2))


if __name__ == '__main__':
    main()
