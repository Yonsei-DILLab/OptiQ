"""Read-only raw-rollout verification; report source is separate from training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,subprocess,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;A=R.parent;W=A.parent/'tmp/reward-progress-worktree'
SHA='b111a993b1e1bf895966dac4469abdfdb2c37c05'
HELPER=A/'antmaze_fixed64_v34_250k/report_results.py'
spec=importlib.util.spec_from_file_location('verified_sigma_helpers',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
base,reward,controls=h.base,h.reward,h.CONTROLS
assert subprocess.check_output(['git','-C',str(W),'show',SHA+':antmaze_experiments/progress_reward.py'])==h.rbytes
read=lambda p:json.loads(p.read_text())
conditions=('capm1','cap0-initm1','cap1-initm1','cap2-initm1','cap3-initm1','uncapped-initm1')
caps=dict(zip(conditions,(-1.,0.,1.,2.,3.,math.inf)))
modes=('policy','native');runs={};failures=[]
for task,(control_root,control_sha,floor) in controls.items():
    current=R/'results'/('vast-heechan-180' if task=='v3' else 'vast-heechan-199')
    for condition in conditions:
        host,source=(control_root,control_sha) if condition=='capm1' else (current,SHA)
        manifest=read(host/'manifest.json');assert manifest['source_commit']==source
        job=next(j for j in manifest['jobs'] if j['task']==task and j['teacher_std_floor']==floor
                 and (condition=='capm1' or j.get('actor_sigma_profile')==condition))
        run=runs[task,condition]=dict(job=job,evaluations={m:[] for m in modes},verified_final=False)
        folder=host/'runs'/job['id']
        if not (folder/'config.json').exists():
            if condition!='capm1':
                status=read(host/'jobs'/(job['id']+'.json'))
                failures.append(dict(task=task,condition=condition,status=status,main_training_started=False))
            continue
        cfg=read(folder/'config.json');actor=cfg['native']['alg']['actor']
        assert cfg['source_commit']==source and cfg['seed']==0 and cfg['task']==task
        expected_profile=reward.START_NORMALIZED_PROFILE if task=='v3' else reward.NO_COST_PROFILE
        assert cfg['reward_profile']==expected_profile and cfg['reward_specification']==reward.specification(task,expected_profile)
        assert cfg['temperature']==(3. if task=='v3' else 1.) and cfg['native']['alg']['gamma']==.999
        assert cfg['dacer_enabled'] and not cfg['noveld_enabled']
        assert cfg['dacer_target_entropy_per_dim']==.7 and cfg['dacer_interval_updates']==500
        assert cfg['effective_eval_starts']=='fixed' and cfg['eval_starts']=='upstream'
        assert cfg['steps']==258304 and cfg['expected_updates']==7816
        assert (cfg['num_envs'],cfg['updates_per_vector_step'],cfg['batch_size'])==(256,8,4096)
        assert actor['teacher_std_floor']==floor and actor.get('latent_prior','normal')=='normal'
        assert (actor['log_std_min'],actor['log_std_max'],actor['initial_log_std'],actor['num_policy_samples'])==(-5.,caps[condition],-1.,64)
        if condition!='capm1':
            initial=read(folder/'actor-sigma-initial-verification.json')
            assert initial['verified'] and initial['parameters']==manifest['initial_parameter_control']
            assert all(c['finite_marginal_nll_and_gradients'] for c in initial['scratch_bound_checks'])
        for mode in modes:
            for path in sorted((folder/'evaluations').glob(f'*/{mode}-fixed/summary.json')):
                if not path.with_name('rollouts.npz').exists():continue
                e=base.evaluate(path,cfg,condition,mode)
                row=e['row'];assert row['episodes']==(100 if row['step']==258304 else 40)
                assert row['step'] in (50176,100096,150016,200192,250112,258304)
                row.update(actor_log_std_cap=caps[condition],initial_log_std=-1.,conditional_sigma=mode=='policy',latent_prior='normal')
                run['evaluations'][mode].append(e)
        if (folder/'result.json').exists():
            result=read(folder/'result.json')
            assert result['completed'] and result['source_commit']==source
            assert (result['steps'],result['updates'])==(258304,7816)
            assert all(result['checkpoint'][k] for k in ('readback_verified','environment_reward_verified','progress_replay_verified'))
            assert (result['checkpoint']['replay_count'],result['checkpoint']['simulator_count'])==(258304,256)
            if condition!='capm1':
                v=read(folder/'actor-sigma-final-verification.json')
                assert v['verified'] and v['profile']==condition and v['actor_updates']==v['critic_updates']==7816
                assert v['observed_log_std_max']<=caps[condition]+1e-6
                run['last_minibatch_actor_std_mean']=result['teacher_update_verification']['actor_std_mean']
                run['training_successes']=read(folder/'training-successes.json')
            for mode in modes:assert run['evaluations'][mode][-1]['row']['step']==258304
            run['verified_final']=True
out=R/'report';out.mkdir(exist_ok=True)
for task in controls:
    for mode in modes:
        fig,axes=plt.subplots(2,3,figsize=(15,10.8))
        fig.subplots_adjust(left=.06,right=.98,bottom=.055,top=.85,wspace=.24,hspace=.4)
        for ax,c in zip(axes.flat,conditions):
            es=runs[task,c]['evaluations'][mode];e=es[-1] if es else None
            label='No upper cap' if math.isinf(caps[c]) else f'log sigma upper {caps[c]:g}'
            base.draw(ax,task,e,label)
            if e is None:
                ax.set_title('No upper cap\nConfig serialization failed before training\nNo rollout result',fontsize=10)
        noise='random z + conditional sigma' if mode=='policy' else 'random z, mu-only supplement'
        reward_name='normalized remaining geodesic; T3; teacher floor1' if task=='v3' else 'geodesic; T1; teacher floor0.5'
        fig.suptitle(f'{task} | {noise} | same initial full state | seed0\nAll completed panels: 250k postwarmup, 100 episodes; init log sigma -1\n{reward_name}; gamma .999; DACER H/d +.7; no bonus/step cost/NovelD',fontsize=11)
        fig.savefig(out/f'{task}_final_{mode}.png',dpi=160);plt.close(fig)
for mode in modes:
    fig,axes=plt.subplots(2,3,figsize=(14,7.2),layout='constrained')
    for i,task in enumerate(controls):
        for c in conditions[:-1]:
            rows=[e['row'] for e in runs[task,c]['evaluations'][mode]]
            for ax,key in zip(axes[i],('success_rate','minority_entry_rate','minority_success_rate')):
                ax.plot([(r['step']-8192)/1000 for r in rows],[r[key]*100 for r in rows],marker='.',label=f'cap {caps[c]:g}')
        for ax,title in zip(axes[i],('Any-goal success (%)','Less-used first gate (%)','Less-used successful route (%)')):
            ax.set_title(task+' | '+title);ax.set_xlabel('Postwarmup transitions (k)');ax.grid(alpha=.2)
        axes[i,0].legend(fontsize=8)
    fig.suptitle(f'Sigma upper-bound grid | {mode} | seed0\n40 episodes per intermediate evaluation, 100 at final; uncapped failed before training',fontsize=11)
    fig.savefig(out/f'learning_curves_{mode}.png',dpi=160);plt.close(fig)
history=[e['row'] for r in runs.values() for es in r['evaluations'].values() for e in es]
latest=[es[-1]['row'] for r in runs.values() for es in r['evaluations'].values() if es]
payload=dict(time_utc=datetime.now(timezone.utc).isoformat(),training_source=SHA,
             control_sources={t:v[1] for t,v in controls.items()},history=history,latest=latest,failures=failures,
             final_verified=[dict(task=t,condition=c,id=r['job']['id']) for (t,c),r in runs.items() if r['verified_final']],
             last_minibatch_actor_std_mean={t:{c:runs[t,c].get('last_minibatch_actor_std_mean') for c in conditions[:-1]} for t in controls},
             report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
             common_default_profile=False,training_changed=False,goal_retention_established=False)
(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
lines=['# σ 상한 비교 — 250k 결과','',
'유한 상한0/1/2/3의8개 완료. 무상한2개는 inf를 JSON으로 저장하는 단계에서 실패했고 본학습을 시작하지 못했다. 학습 발산으로 분류하지 않는다.','',
'아래는 같은 초기 full state에서 매 행동 random z와 conditional sigma를 샘플링한 직접 정책100회 결과다. 초기log sigma는 모두-1이며 상한-1은 보존 대조군이다.','',
'|환경|log σ 상한|평가|통로 진입|성공 통로|성공률|','|---|---|---|---|---|---:|']
for row in latest:
    lines.append(f"|{row['task']}|{row['actor_log_std_cap']:g}|{row['mode']}|{row['route_counts']}|{row['successful_route_counts']}|{row['success_rate']:.0%}|")
lines+=['','![v3 직접정책](v3_final_policy.png)','![v4 직접정책](v4_final_policy.png)','![학습곡선](learning_curves_policy.png)','',
'이 실험은 공통 기본값이 아니다. v3=정규화된geodesic progress100/T3/teacherfloor1, v4=geodesic progress100/T1/teacherfloor.5, 둘 다gamma.999/DACER목표+.7·500update/NovelD OFF이다. 각 환경 내에서만 상한 효과를 비교한다.',
'단일seed·250k이다. 모든실패를 분모에 포함하고 진입과 성공을 구분했다. 정책·학습코드는 바꾸지 않았으며 원시rollout의SHA, 시작상태, 보상, 목표도달과 최종replay검증기록을 대조했다. 무상한의 성능 결과는 없다.']
(out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(report=str(out),verified_final=len(payload['final_verified']),failed_before_training=len(failures),latest=[r for r in latest if r['mode']=='policy']),ensure_ascii=False))
