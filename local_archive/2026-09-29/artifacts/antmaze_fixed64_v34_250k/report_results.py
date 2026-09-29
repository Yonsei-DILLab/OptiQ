"""Matched-step random/finite prior comparison from immutable raw evaluations."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
A = ROOT.parent
W = A.parent/'tmp/reward-progress-worktree'
SOURCE = '9b733bad9d52ba73fad01f03521f18a211d21a1e'
CONTROLS = {
    'v3': (A/'antmaze_teacher_floor_250k_r2/results/vast-heechan-180',
           'd25930197ee9ba060a3aa84c3b6fea419dfe856d', 1.),
    'v4': (A/'antmaze_v4_teacher_floor_250k/results/vast-heechan-199',
           '555bb7e3c0101ee939545cc35d4d0729291781ec', .5),
}
HELPER = A/'antmaze_startnorm_geodesic_250k/report_results.py'
spec = importlib.util.spec_from_file_location('fixed64_reward_report', HELPER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base, reward = m.base, m.reward
rbytes = subprocess.check_output(['git', '-C', str(W), 'show',
                                 SOURCE+':antmaze_experiments/progress_reward.py'])
assert rbytes == m.reward_bytes
base.reward = reward
OUT = ROOT/'report'
MODES = ('policy', 'native')
CONDITIONS = ('random-prior', 'fixed64-prior')
read = lambda p: json.loads(p.read_text())


def collect():
    runs = {}
    finite_host = ROOT/'results/vast-heechan-199'
    for task, (control_host, control_source, floor) in CONTROLS.items():
        for host, source, condition in (
                (control_host, control_source, 'random-prior'),
                (finite_host, SOURCE, 'fixed64-prior')):
            manifest = read(host/'manifest.json')
            assert manifest['source_commit'] == source
            job = next(j for j in manifest['jobs'] if j['task'] == task
                       and j.get('teacher_std_floor') == floor)
            run = dict(job=job, source=source, condition=condition,
                       evaluations={mode: [] for mode in MODES}, verified_final=False)
            runs[task, condition] = run
            folder = host/'runs'/job['id']
            if not (folder/'config.json').exists():
                continue
            cfg = read(folder/'config.json')
            run['config'] = cfg
            assert cfg['source_commit'] == source and cfg['seed'] == 0 and cfg['task'] == task
            profile = reward.START_NORMALIZED_PROFILE if task == 'v3' else reward.NO_COST_PROFILE
            assert cfg['reward_profile'] == profile
            assert cfg['reward_specification'] == reward.specification(task, profile)
            assert cfg['temperature'] == (3. if task == 'v3' else 1.)
            assert cfg['native']['alg']['gamma'] == .999
            assert cfg['dacer_enabled'] and not cfg['noveld_enabled']
            assert cfg['dacer_target_entropy_per_dim'] == .7 and cfg['dacer_interval_updates'] == 500
            assert cfg['eval_starts'] == 'upstream' and cfg['effective_eval_starts'] == 'fixed'
            actor = cfg['native']['alg']['actor']
            assert actor['teacher_std_floor'] == floor and cfg['teacher_std_floor_override'] == floor
            assert (actor['log_std_min'], actor['log_std_max'], actor['num_policy_samples']) == (-5., -1., 64)
            if condition == 'fixed64-prior':
                assert cfg['latent_profile'] == 'fixed64' and actor['latent_prior'] == 'finite'
                assert actor['latent_components'] == 64 and actor['latent_codebook_seed'] == 20260911
                initial = read(folder/'latent-profile-initial-verification.json')
                assert initial['verified'] and initial['actor_updates'] == 0
                codes = np.load(folder/'latent-codebook.npy', allow_pickle=False)
                assert codes.shape == (64, 8)
                assert hashlib.sha256(codes.tobytes()).hexdigest() == initial['codebook_sha256']
                assert initial['direct_sampler_verified'] and initial['native_sampler_verified']
            else:
                assert actor.get('latent_prior', 'normal') == 'normal'
            for mode in MODES:
                for path in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
                    if path.with_name('rollouts.npz').exists():
                        evaluation = base.evaluate(path, cfg, condition, mode)
                        evaluation['row']['latent_prior'] = 'finite64' if condition == 'fixed64-prior' else 'normal'
                        evaluation['row']['conditional_sigma'] = mode == 'policy'
                        assert evaluation['row']['episodes'] == (100 if evaluation['row']['step'] == 258304 else 40)
                        run['evaluations'][mode].append(evaluation)
            if (folder/'result.json').exists():
                proof = read(folder/'result.json')
                assert proof['completed'] and proof['source_commit'] == source
                assert proof['steps'] == 258304 and proof['updates'] == 7816
                assert proof['checkpoint']['environment_reward_verified']
                assert proof['checkpoint']['progress_replay_verified'] and proof['checkpoint']['readback_verified']
                assert proof['checkpoint']['replay_count'] == 258304
                if condition == 'fixed64-prior':
                    final = read(folder/'latent-profile-final-verification.json')
                    assert final['verified'] and final['actor_updates'] == 7816
                    assert final['codebook_sha256'] == initial['codebook_sha256']
                    assert set(proof['summaries']) == {'native-fixed', 'policy-fixed', 'component0_mu-fixed'}
                for mode in MODES:
                    assert run['evaluations'][mode][-1]['row']['step'] == 258304
                run['verified_final'] = True
    return runs


def main():
    OUT.mkdir(exist_ok=True)
    runs = collect()
    history = [e['row'] for run in runs.values() for es in run['evaluations'].values() for e in es]
    latest = [es[-1]['row'] for run in runs.values() for es in run['evaluations'].values() if es]
    matched = []
    for mode in MODES:
        fig, axes = plt.subplots(2, 2, figsize=(11, 11.5), layout='constrained')
        for i, task in enumerate(CONTROLS):
            by_condition = {c: {e['row']['step']: e for e in runs[task, c]['evaluations'][mode]}
                            for c in CONDITIONS}
            common = sorted(set.intersection(*(set(v) for v in by_condition.values())))
            step = common[-1] if common else None
            for k, condition in enumerate(CONDITIONS):
                evaluation = by_condition[condition].get(step)
                base.draw(axes[i, k], task, evaluation, f'{task} | {condition}')
                if evaluation:
                    matched.append(evaluation['row'])
        noise = 'Conditional sigma INCLUDED' if mode == 'policy' else 'Mu-only supplement'
        fig.suptitle(f'Latent prior comparison | {mode}: {noise}\n'
                     'Same full origin, seed0; same step within each row; no external noise\n'
                     'v3: normalized geodesic / T3 / teacher floor1; v4: geodesic / T1 / floor0.5', fontsize=11)
        fig.savefig(OUT/f'matched_trajectories_{mode}.png', dpi=155)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), layout='constrained')
        for i, task in enumerate(CONTROLS):
            for condition in CONDITIONS:
                rows = [e['row'] for e in runs[task, condition]['evaluations'][mode]]
                if not rows:
                    continue
                x = [r['step']/1000 for r in rows]
                for ax, key, scale in zip(axes[i], ('success_rate', 'minority_entry_rate', 'minority_success_rate'), (100, 100, 100)):
                    ax.plot(x, [scale*r[key] for r in rows], marker='.', label=condition)
            for ax, title in zip(axes[i], ('Any-goal success (%)', 'Less-used first gate (%)', 'Less-used successful route (%)')):
                ax.set_title(f'{task} | {title}'); ax.set_xlabel('Total transitions (k)'); ax.grid(alpha=.2)
                ax.set_ylim(-2, 102 if 'Any-goal' in title else 52)
            axes[i, 0].legend(fontsize=8)
        fig.suptitle(f'{mode} | failures included; fixed64 resamples a component every action; one training seed', fontsize=11)
        fig.savefig(OUT/f'learning_curves_{mode}.png', dpi=155)
        plt.close(fig)
    final = [dict(task=t, condition=c, id=r['job']['id']) for (t,c),r in runs.items() if r['verified_final']]
    payload = dict(time_utc=datetime.now(timezone.utc).isoformat(), source_commit=SOURCE,
                   control_sources={t:v[1] for t,v in CONTROLS.items()}, final_verified=final,
                   retention_established=False, history=history, latest=latest, exact_step_matched=matched,
                   report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                   reward_implementation_sha256=hashlib.sha256(rbytes).hexdigest())
    (OUT/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    lines = ['# 고정 latent 64개와 기존 random latent 비교', '',
             '각 환경의 대조군과 보상·temperature·teacher floor·모델·초기 가중치·학습 예산은 같습니다. '
             '기존 코드에 있는 finite64 설정만 사용합니다. 매 행동마다 동일한 64개 codebook에서 균등하게 다시 선택하며, episode 동안 z를 고정하지 않습니다.', '',
             '|환경|prior|mode|total step|평가 수|통로 진입|성공 통로|',
             '|---|---|---|---:|---:|---|---|']
    for r in latest:
        lines.append(f"|{r['task']}|{r['condition']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|")
    lines += ['', '![동일 step 직접 정책 비교](matched_trajectories_policy.png)',
              '![직접 정책 학습 추이](learning_curves_policy.png)', '',
              '그림은 각 환경에서 두 조건의 동일한 step을 비교합니다. 표는 조건별 최신 평가로 step이 다를 수 있습니다. '
              'v3·v4 모두 원래 동일한 위치·자세·속도에서 시작합니다. 직접 정책은 conditional sigma를 포함하며 외부 DACER 잡음을 넣지 않습니다. '
              'mu-only는 별도 보조 결과입니다. 첫 통로 진입 수와 해당 episode의 목표 성공 수를 분리합니다. '
              '실패를 분모에서 제외하지 않으며 한 seed의 250k 결과만으로 장기 유지나 재현성을 주장하지 않습니다.', '',
              '실제 8update 사전검사, sampling/serialization, 초기 parameter hash, 고정 codebook, frozen 알고리즘 파일 9개를 검사했습니다. '
              '보고서는 raw 궤적의 full start·보상합·성공 목표·SHA256을 검증하며 학습 결과를 덮어쓰지 않습니다. '
              'finite의 결정론적 보조 평가는 component0_mu로 저장되며 z=0 평가로 부르지 않습니다.', '',
              '완료 및 최종 replay 검증된 조건: '+str(final),
              '보고 코드는 학습 시작 후 만들어진 후처리이며 SHA와 학습 source를 results.json에 구분했습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT), rows=len(history), final_verified=final,
                         matched=[{k:r[k] for k in ('task','condition','mode','step','route_counts','successful_route_counts')}
                                  for r in matched])))


if __name__ == '__main__':
    main()
