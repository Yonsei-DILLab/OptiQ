"""Post-hoc matched-budget sigma-cap report with raw rollout validation."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
A=ROOT.parent
W=A.parent/'tmp/reward-progress-worktree'
SOURCE='6e9090c8b9bb0ed7d1700ab6339a481bc97bf84d'
HELPER=A/'antmaze_fixed64_v34_250k/report_results.py'
spec=importlib.util.spec_from_file_location('sigma_raw_helpers',HELPER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base,reward,CONTROLS=m.base,m.reward,m.CONTROLS
rbytes=subprocess.check_output(['git','-C',str(W),'show',SOURCE+':antmaze_experiments/progress_reward.py'])
assert rbytes==m.rbytes
OUT=ROOT/'report'
CONDITIONS=('capm1','capm2','capm3')
MODES=('policy','native')
read=lambda p:json.loads(p.read_text())


def collect():
    runs={}
    current_host=ROOT/'results/vast-heechan-199'
    for task,(control_host,control_source,floor) in CONTROLS.items():
        for condition in CONDITIONS:
            host,source=(control_host,control_source) if condition=='capm1' else (current_host,SOURCE)
            manifest=read(host/'manifest.json')
            assert manifest['source_commit']==source
            job=next(j for j in manifest['jobs'] if j['task']==task and j.get('teacher_std_floor')==floor
                     and (condition=='capm1' or j.get('actor_sigma_profile')==condition))
            run=dict(job=job,source=source,evaluations={mode:[] for mode in MODES},verified_final=False)
            runs[task,condition]=run
            folder=host/'runs'/job['id']
            if not (folder/'config.json').exists():continue
            cfg=read(folder/'config.json')
            cap=-float(condition[-1])
            assert cfg['source_commit']==source and cfg['seed']==0 and cfg['task']==task
            profile=reward.START_NORMALIZED_PROFILE if task=='v3' else reward.NO_COST_PROFILE
            assert cfg['reward_profile']==profile and cfg['reward_specification']==reward.specification(task,profile)
            assert cfg['temperature']==(3. if task=='v3' else 1.) and cfg['native']['alg']['gamma']==.999
            assert cfg['dacer_enabled'] and not cfg['noveld_enabled']
            assert cfg['dacer_target_entropy_per_dim']==.7 and cfg['dacer_interval_updates']==500
            assert cfg['eval_starts']=='upstream' and cfg['effective_eval_starts']=='fixed'
            assert cfg['steps']==258304 and cfg['expected_updates']==7816
            assert cfg['num_envs']==256 and cfg['updates_per_vector_step']==8 and cfg['batch_size']==4096
            assert cfg['warmup_transitions']==8192 and cfg['updates_per_transition']==1/32
            actor=cfg['native']['alg']['actor']
            assert actor['teacher_std_floor']==floor and actor.get('latent_prior','normal')=='normal'
            assert (actor['log_std_min'],actor['log_std_max'],actor['initial_log_std'],actor['num_policy_samples'])==(-5.,cap,cap,64)
            if condition!='capm1':
                assert cfg['actor_sigma_profile']==condition
                initial=read(folder/'actor-sigma-initial-verification.json')
                assert initial['verified'] and initial['actor_updates']==initial['critic_updates']==0
                assert initial['control_actor_after_restoring_only_initial_sigma_bias']==manifest['initial_parameter_control']['actor']
                assert initial['parameters']['critic']==manifest['initial_parameter_control']['critic']
            for mode in MODES:
                for path in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
                    if not path.with_name('rollouts.npz').is_file():continue
                    e=base.evaluate(path,cfg,condition,mode)
                    assert e['row']['step'] in (50176,100096,150016,200192,250112,258304)
                    assert e['row']['episodes']==(100 if e['row']['step']==258304 else 40)
                    e['row'].update(actor_log_std_cap=cap,actor_initial_log_std=cap,
                                    conditional_sigma=mode=='policy',latent_prior='normal')
                    run['evaluations'][mode].append(e)
            if (folder/'result.json').exists():
                proof=read(folder/'result.json')
                assert proof['completed'] and proof['source_commit']==source
                assert proof['steps']==258304 and proof['updates']==7816
                assert all(proof['checkpoint'][k] for k in ('readback_verified','environment_reward_verified','progress_replay_verified'))
                assert proof['checkpoint']['replay_count']==258304 and proof['checkpoint']['simulator_count']==256
                if condition!='capm1':
                    verification=read(folder/'actor-sigma-final-verification.json')
                    assert verification['verified'] and verification['profile']==condition
                    assert verification['actor_updates']==verification['critic_updates']==7816
                    assert verification['observed_log_std_max']<=cap+1e-6
                    assert verification['model_optimizer_rng_unchanged']
                for mode in MODES:assert run['evaluations'][mode][-1]['row']['step']==258304
                run['verified_final']=True
    return runs


def main():
    OUT.mkdir(exist_ok=True)
    runs=collect()
    history=[e['row'] for r in runs.values() for es in r['evaluations'].values() for e in es]
    latest=[es[-1]['row'] for r in runs.values() for es in r['evaluations'].values() if es]
    matched=[]
    for mode in MODES:
        fig,axes=plt.subplots(2,3,figsize=(16,11.5),layout='constrained')
        for i,task in enumerate(CONTROLS):
            by={c:{e['row']['step']:e for e in runs[task,c]['evaluations'][mode]} for c in CONDITIONS}
            common=sorted(set.intersection(*(set(v) for v in by.values())))
            step=common[-1] if common else None
            for k,c in enumerate(CONDITIONS):
                e=by[c].get(step);base.draw(axes[i,k],task,e,f'{task} | log sigma cap/init {-int(c[-1])}')
                if e:matched.append(e['row'])
        fig.suptitle(f'Conditional sigma cap AND initialization | {mode}\n'
            'Same full origin and total step; random latent; seed0; direct includes trained conditional sigma\n'
            'v3: normalized geodesic / T3 / teacher floor1; v4: geodesic / T1 / teacher floor0.5',fontsize=12)
        fig.savefig(OUT/f'matched_trajectories_{mode}.png',dpi=150);plt.close(fig)
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
        fig.suptitle(f'{mode} | conditional sigma cap and initialization varied together; failures included')
        fig.savefig(OUT/f'learning_curves_{mode}.png',dpi=150);plt.close(fig)
    final=[dict(task=t,condition=c,id=r['job']['id']) for (t,c),r in runs.items() if r['verified_final']]
    payload=dict(time_utc=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE,
        control_sources={t:v[1] for t,v in CONTROLS.items()},final_verified=final,
        retention_established=False,history=history,latest=latest,exact_step_matched=matched,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),reward_sha256=hashlib.sha256(rbytes).hexdigest())
    (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Conditional sigma 상한·초기값 비교','',
        'log sigma 상한과 초기값을 함께 −1/−2/−3으로 비교합니다. 하한−5, random latent, teacher 탐색폭, '
        '보상·모델·학습률·총 transition·업데이트 수는 같습니다. 상한 효과와 초기화 효과를 분리한 실험은 아닙니다.','',
        '|환경|조건|mode|total step|평가 수|통로 진입|성공 통로|',
        '|---|---|---|---:|---:|---|---|']
    for r in latest:
        lines.append(f"|{r['task']}|{r['condition']}|{r['mode']}|{r['step']}|{r['episodes']}|{r['route_counts']}|{r['successful_route_counts']}|")
    lines+=['','![동일 step 직접 정책](matched_trajectories_policy.png)',
        '![직접 정책 학습 추이](learning_curves_policy.png)','',
        '주 그림은 모든 조건의 동일 step만 비교합니다. 표는 조건별 최신 평가입니다. 직접 정책은 학습된 conditional sigma를 '
        '실제로 샘플링하며 외부 DACER 잡음을 넣지 않습니다. mu-only는 보조 결과입니다. '
        'v3/v4는 원래 동일 full state로 시작합니다. 실패를 분모에서 제외하지 않으며, 통로 진입만으로 성공이나 장기 유지를 주장하지 않습니다. '
        '단일 seed이며 원시 궤적의 보상·full start·목표·SHA, 최종 replay와 초기 parameter 동등성을 검증합니다.',
        '최종 검증된 조건: '+str(final),
        '보고 코드는 학습 후 작성한 후처리이며 학습 source와 SHA를 별도로 저장했습니다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT),rows=len(history),final_verified=final)))


if __name__=='__main__':main()
