"""Read-only signed-target, queue and unchanged-control verification."""
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE='4a33d13f0a8de3be508537fd38879ac8ef8018c0'
controls=Path((ROOT.parent/'antmaze_anneal_baselines/latest-report-path.txt').read_text().strip())
reference={}
for task in ('v1','v3','v4'):
    run=next(controls.glob(f'*/antmaze-optiq-dense-off-T1-s0-20260924/runs/{task}-optiq-s0'))
    reference[task]=(json.loads((run/'config.json').read_text()),json.loads((run/'parameter-audit.json').read_text()))
jobs=[];hosts={};passed=[];live_checks=[]
for host in ('vast-heechan-180','vast-heechan-199'):
    data=json.loads((ROOT/(host+'.json')).read_text())
    manifest=data['manifest.json'];status=data['status.json']
    assert manifest['source_commit']==SOURCE==data['branch_head'] and not data['branch_status']
    assert manifest['wandb_mode']=='online' and manifest['wandb_project']=='antmaze'
    assert not status['failed'] and not status['pending_held'] and 'failure.json' not in data
    hosts[host]={k:status[k] for k in ('running','pending','completed','failed')}
    for job in manifest['jobs']:
        jobs.append(job)
        target=job['dacer_target_entropy_per_dim']
        assert job['temperature']==1 and job['reward_profile']=='dense' and job['noveld']=='off'
        for phase in ('preflight','runs'):
            prefix=f'{phase}/{job["id"]}/'
            config=data.get(prefix+'config.json')
            if config is None:continue
            assert config['source_commit']==SOURCE
            assert config['temperature_schedule'] is None
            assert config['dacer_target_entropy_per_dim']==target
            assert config['native']['alg']==reference[job['task']][0]['native']['alg']
            dacer=copy.deepcopy(config['native']['dacer'])
            assert dacer['target_entropy_per_dim']==target
            dacer['target_entropy_per_dim']=-.9
            assert dacer==reference[job['task']][0]['native']['dacer']
            for k in ('num_envs','batch_size','updates_per_vector_step','warmup_transitions','reward_profile','noveld_enabled','eval_starts','eval_interval','interim_eval_episodes','final_eval_episodes','save_intermediate_policy'):
                assert config[k]==reference[job['task']][0][k],k
            regulator=data.get(prefix+'dacer_regulator.json')
            if regulator:
                assert math.isclose(regulator['target_entropy'],8*target,abs_tol=1e-12)
                assert regulator['updates']>=1 and math.isfinite(regulator['entropy_proxy'])
            result=data.get(prefix+'result.json')
            if phase=='preflight' and result:
                assert result['completed'] and result['steps']==8448 and result['updates']==8 and result['rnd_updates']==0
                assert result['checkpoint']['dense_replay_verified']
                param=data[prefix+'parameter-audit.json']['initial']
                for k in ('actor','critic'):assert param[k]==reference[job['task']][1]['initial'][k]
                passed.append(job['id'])
            if phase=='runs' and regulator:
                meta=data.get(prefix+'wandb.json',{})
                assert meta.get('mode')=='online'
                live_checks.append(dict(id=job['id'],target_per_dim=target,total_target=regulator['target_entropy'],
                    updates=regulator['updates'],noise_std=regulator['noise_std'],wandb=meta.get('url')))
assert len(jobs)==15==len({j['id'] for j in jobs})
assert {(j['task'],j['dacer_target_entropy_per_dim']) for j in jobs}=={(t,h) for t in ('v1','v3','v4') for h in (-1.,-.8,-.5,-.3,-.1)}
proof=dict(time=datetime.now(timezone.utc).isoformat(),source=SOURCE,verified=True,
    total_jobs=15,hosts=hosts,preflight_passed=passed,actual_training_target_checks=live_checks,
    control_comparison='Actual native alg and DACER configurations unchanged except requested target; completed preflight initial actor/critic hashes match existing T=1 control.')
(ROOT/'registration-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(total_jobs=15,preflight_passed=len(passed),training_targets_verified=len(live_checks),running=sum(len(h['running']) for h in hosts.values()),pending=sum(len(h['pending']) for h in hosts.values())),indent=2))
