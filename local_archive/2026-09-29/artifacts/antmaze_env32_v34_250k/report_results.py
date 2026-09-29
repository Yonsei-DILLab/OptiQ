"""Collection-count comparison using exact matched transitions and raw rollouts."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
A = ROOT.parent
W = A.parent/'tmp/reward-progress-worktree'
SOURCE = '7d1b9f2ee63ba959e66f3453f2587cdbd6d993a3'
HELPER = A/'antmaze_fixed64_v34_250k/report_results.py'
spec = importlib.util.spec_from_file_location('collection_raw_helpers', HELPER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base, reward, CONTROLS = m.base, m.reward, m.CONTROLS
rbytes = subprocess.check_output(['git','-C',str(W),'show',SOURCE+':antmaze_experiments/progress_reward.py'])
assert rbytes == m.rbytes
OUT = ROOT/'report'
MODES = ('policy','native')
CONDITIONS = ('env256-update8','env32-update1')
read = lambda p: json.loads(p.read_text())


def collect():
    runs = {}
    current_host = ROOT/'results/vast-heechan-180'
    for task, (control_host, control_source, floor) in CONTROLS.items():
        for host, source, condition in ((control_host, control_source, CONDITIONS[0]),
                                        (current_host, SOURCE, CONDITIONS[1])):
            manifest = read(host/'manifest.json')
            assert manifest['source_commit'] == source
            job = next(j for j in manifest['jobs'] if j['task']==task and j.get('teacher_std_floor')==floor)
            run = dict(job=job, source=source, evaluations={mode:[] for mode in MODES}, verified_final=False)
            runs[task, condition] = run
            folder = host/'runs'/job['id']
            if not (folder/'config.json').exists():
                continue
            cfg = read(folder/'config.json')
            assert cfg['source_commit']==source and cfg['seed']==0 and cfg['task']==task
            profile = reward.START_NORMALIZED_PROFILE if task=='v3' else reward.NO_COST_PROFILE
            assert cfg['reward_profile']==profile and cfg['reward_specification']==reward.specification(task,profile)
            assert cfg['temperature']==(3. if task=='v3' else 1.) and cfg['native']['alg']['gamma']==.999
            assert cfg['dacer_enabled'] and not cfg['noveld_enabled']
            assert cfg['dacer_target_entropy_per_dim']==.7 and cfg['dacer_interval_updates']==500
            assert cfg['eval_starts']=='upstream' and cfg['effective_eval_starts']=='fixed'
            assert cfg['steps']==258304 and cfg['expected_updates']==7816 and cfg['batch_size']==4096
            assert cfg['warmup_transitions']==8192 and cfg['updates_per_transition']==1/32
            actor = cfg['native']['alg']['actor']
            assert actor['teacher_std_floor']==floor and actor.get('latent_prior','normal')=='normal'
            assert (actor['log_std_min'],actor['log_std_max'],actor['num_policy_samples'])==(-5.,-1.,64)
            if condition==CONDITIONS[1]:
                assert cfg['collection_profile']=='env32-update1'
                assert cfg['num_envs']==32 and cfg['updates_per_vector_step']==1
                initial = read(folder/'collection-profile-initial-verification.json')
                assert initial['verified'] and initial['actor_updates']==initial['critic_updates']==0
                assert initial['actual_envs']==32 and initial['actual_updates_per_collection']==1
            else:
                assert cfg['num_envs']==256 and cfg['updates_per_vector_step']==8
            for mode in MODES:
                for path in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
                    if not path.with_name('rollouts.npz').is_file():
                        continue
                    e = base.evaluate(path,cfg,condition,mode)
                    assert e['row']['step'] in (50176,100096,150016,200192,250112,258304)
                    assert e['row']['episodes']==(100 if e['row']['step']==258304 else 40)
                    e['row'].update(num_envs=cfg['num_envs'],updates_per_vector_step=cfg['updates_per_vector_step'],
                                    learner_updates=(e['row']['step']-8192)//32,
                                    conditional_sigma=mode=='policy',latent_prior='normal')
                    run['evaluations'][mode].append(e)
            if (folder/'result.json').exists():
                proof = read(folder/'result.json')
                assert proof['completed'] and proof['source_commit']==source
                assert proof['steps']==258304 and proof['updates']==7816
                assert all(proof['checkpoint'][k] for k in ('readback_verified','environment_reward_verified','progress_replay_verified'))
                assert proof['checkpoint']['replay_count']==258304
                assert proof['checkpoint']['simulator_count']==cfg['num_envs']
                if condition==CONDITIONS[1]:
                    verification = read(folder/'collection-profile-final-verification.json')
                    assert verification['verified'] and verification['actual_updates']==7816
                    assert verification['env_steps_each']==8072 and verification['simulator_count']==32
                    assert verification['expected_eval_steps']==[50176,100096,150016,200192,250112]
                for mode in MODES:
                    assert run['evaluations'][mode][-1]['row']['step']==258304
                run['verified_final']=True
    return runs


def main():
    OUT.mkdir(exist_ok=True)
    runs=collect()
    history=[e['row'] for r in runs.values() for es in r['evaluations'].values() for e in es]
    latest=[es[-1]['row'] for r in runs.values() for es in r['evaluations'].values() if es]
    matched=[]
    for mode in MODES:
        fig,axes=plt.subplots(2,2,figsize=(11,11.5),layout='constrained')
        for i,task in enumerate(CONTROLS):
            by={c:{e['row']['step']:e for e in runs[task,c]['evaluations'][mode]} for c in CONDITIONS}
            common=sorted(set.intersection(*(set(v) for v in by.values())))
            step=common[-1] if common else None
            for k,c in enumerate(CONDITIONS):
                e=by[c].get(step);base.draw(axes[i,k],task,e,f'{task} | {c}')
                if e:matched.append(e['row'])
        fig.suptitle(f'Collection concurrency | {mode} | global update ratio 1/32, batch4096\n'
                     'Same full origin and global step; random latent; seed0; direct includes conditional sigma\n'
                     'v3: normalized geodesic / T3 / floor1; v4: geodesic / T1 / floor0.5',fontsize=11)
        fig.savefig(OUT/f'matched_trajectories_{mode}.png',dpi=155);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,7.4),layout='constrained')
        for i,task in enumerate(CONTROLS):
            for c in CONDITIONS:
                rows=[e['row'] for e in runs[task,c]['evaluations'][mode]]
                if not rows:continue
                x=[r['step']/1000 for r in rows]
                for ax,key in zip(axes[i],('success_rate','minority_entry_rate','minority_success_rate')):
                    ax.plot(x,[100*r[key] for r in rows],marker='.',label=c)
            for ax,title in zip(axes[i],('Any-goal success (%)','Less-used first gate (%)','Less-used successful route (%)')):
                ax.set_title(f'{task} | {title}');ax.set_xlabel('Total transitions (k)');ax.grid(alpha=.2)
            axes[i,0].legend(fontsize=8)
        fig.suptitle(f'{mode} | same total updates, different collection/update block sizes; failures included',fontsize=11)
        fig.savefig(OUT/f'learning_curves_{mode}.png',dpi=155);plt.close(fig)
    final=[dict(task=t,condition=c,id=r['job']['id']) for (t,c),r in runs.items() if r['verified_final']]
    payload=dict(time_utc=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE,
        control_sources={t:v[1] for t,v in CONTROLS.items()},final_verified=final,
        retention_established=False,history=history,latest=latest,exact_step_matched=matched,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),reward_sha256=hashlib.sha256(rbytes).hexdigest())
    (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# 병렬 환경 수 비교: 총 업데이트 비율 유지','',
        '256개 수집 후 8회 업데이트와 32개 수집 후 1회 업데이트를 비교합니다. '
        '총 transition·업데이트 수·batch4096·모델·초기 가중치·보상·평가 시점은 같고, 병렬 수집과 업데이트 블록 크기가 달라집니다. '
        '각 환경은 최종 예산에서 1,009회 또는 8,072회 step을 수행하므로 replay 구성과 각 궤적 중 정책 변화 속도는 다릅니다.','',
        '|환경|조건|mode|total step|평가 수|통로 진입|성공 통로|',
        '|---|---|---|---:|---:|---|---|']
    for r in latest:
        lines.append(f"|{r['task']}|{r['condition']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|")
    lines+=['','![동일 step 직접 정책](matched_trajectories_policy.png)',
        '![직접 정책 학습 추이](learning_curves_policy.png)','',
        'v3·v4는 원래 동일한 full state에서 시작하며, 직접 정책은 random latent와 conditional sigma를 포함합니다. '
        '외부 DACER 행동잡음은 평가하지 않습니다. mu-only는 보조 결과입니다. '
        '한 seed이며, 통로 진입만으로 성공 경로 다양성이나 장기 유지를 주장하지 않습니다. '
        '원자료의 보상 합·시작 상태·목표 도달·SHA와 최종 replay를 검증합니다.',
        '검증된 최종 조건: '+str(final),
        '보고 코드는 학습 후 작성한 후처리이며 SHA와 학습 source를 results.json에 구분합니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT),rows=len(history),final_verified=final,
        matched=[{k:r[k] for k in ('task','condition','mode','step','route_counts','successful_route_counts')} for r in matched])))


if __name__=='__main__':
    main()
