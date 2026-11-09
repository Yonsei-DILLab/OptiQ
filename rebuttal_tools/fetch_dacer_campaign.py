"""Supervise an iBOLT-only DACER search, fixed-budget confirmation and SAC comparison."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from .fetch_dacer_worker import TARGETS,NOISE_SCALES,verify_saved_configuration
from .fetch_wide_utd1_worker import atomic
from .fetch_wide_utd1_campaign import process_info


def job_spec(plan,target,noise,seed,phase='search'):
    ident=f'pick-and-place-ibolt-dacer-H{target:g}-L{noise:g}-s{seed}'
    return dict(id=ident,method='ibolt',task='pick-and-place',temperature=.25,seed=seed,
        target_entropy_per_dim=target,noise_scale=noise,phase=phase,source=plan['source'],
        source_manifest=plan['source_manifest'],reference_learner_config=plan['reference_learner_config'],
        reference_run_config=plan['reference_run_config'],output=str(Path(plan['folder'])/'runs'/ident),
        status='queued',gpu=None,external=False)


def records(path,step_field='steps'):
    data={}
    for line in path.read_text().splitlines():
        if line.strip():
            row=json.loads(line);data[row[step_field]]=row
    return data


def score_job(job):
    out=Path(job['output']);evaluations=records(out/'evaluations.jsonl')
    es=[evaluations[s] for s in sorted(evaluations) if 0<s<1000000]
    assert len(es)==199 and all(r[r['primary_mode']]['episodes']==10 for r in es)
    late=es[-5:];heights=records(out/'height-evaluations.jsonl')
    air=[heights[r['steps']]['modes']['stochastic_z']['air_over5cm'] for r in late]
    final=evaluations[1000000];assert final[final['primary_mode']]['episodes']==100
    return dict(seed=job['seed'],last5_success=statistics.mean(r[r['primary_mode']]['success_rate'] for r in late),
        auc=statistics.mean(r[r['primary_mode']]['success_rate'] for r in es),
        air_successes=sum(r['final_successes'] for r in air),air_episodes=sum(r['episodes'] for r in air),
        new_successes=sum(r[r['primary_mode']]['new_final_success_count'] for r in late),
        final100_success=final[final['primary_mode']]['success_rate'],
        final100_height=heights[1000000]['modes']['stochastic_z'])


def choose_parameters(jobs,profiles=None):
    profiles=profiles or [(h,l) for h in TARGETS for l in NOISE_SCALES]
    assert len(jobs)==2*len(profiles) and all(j['status']=='completed' for j in jobs)
    assert {(j['target_entropy_per_dim'],j['noise_scale']) for j in jobs}==set(map(tuple,profiles))
    scores=[]
    for target,noise in profiles:
        selected=[j for j in jobs if (j['target_entropy_per_dim'],j['noise_scale'])==(target,noise)]
        assert sorted(j['seed'] for j in selected)==[0,1]
        seeds=[score_job(j) for j in sorted(selected,key=lambda j:j['seed'])]
        n=sum(r['air_episodes'] for r in seeds);hits=sum(r['air_successes'] for r in seeds)
        # Final 100-episode scores remain unavailable to the selection key.
        scores.append(dict(target_entropy_per_dim=target,noise_scale=noise,
            last5_success=statistics.mean(r['last5_success'] for r in seeds),
            auc=statistics.mean(r['auc'] for r in seeds),air_success_rate=hits/n if n else 0.,
            air_successes=hits,air_episodes=n,
            seeds=[{k:v for k,v in r.items() if not k.startswith('final100')} for r in seeds]))
    best=max(scores,key=lambda r:(r['last5_success'],r['air_success_rate'],r['auc'],
        -abs(r['target_entropy_per_dim']+.9),-abs(math.log(r['noise_scale']/.1))))
    return dict(parameters={k:best[k] for k in ('target_entropy_per_dim','noise_scale')},
        selected_unix=time.time(),scores=scores,selection_seeds=[0,1],additional_seeds=[2,3,4],
        evidence_of_high_air_success=best['air_successes']>0,
        rule='Mean of within-seed last five pre-final 10-episode evaluations across seeds 0/1; ties by high-air success, AUC, then proximity to default DACER parameters. Final 100-episode evaluations excluded.',
        limitation='Seeds 0/1 participated in tuning. Seeds 2/3/4 provide additional training-seed confirmation. A best rank alone does not demonstrate learning to lift.')


def aggregate(values):
    return dict(mean=statistics.mean(values),seed_sd=statistics.stdev(values) if len(values)>1 else None,values=values)


def compare_to_sac(jobs,reference,selection):
    chosen=selection['parameters'];group=[j for j in jobs if all(j[k]==v for k,v in chosen.items())]
    assert len(group)==5 and sorted(j['seed'] for j in group)==list(range(5))
    rows=[score_job(j) for j in sorted(group,key=lambda j:j['seed'])];sac=sorted(reference['seeds'],key=lambda r:r['seed'])
    result=dict(checked_unix=time.time(),complete=True,selected_parameters=chosen,iBOLT={},SAC={},paired_gaps={},
        seed_results=rows,selection_seeds=[0,1],additional_confirmation_seeds=[2,3,4],
        fixed_settings='T=.25, logstd[-5,+1], initial=-1, env1, UTD1, batch256, HER+observation normalization, 1M transitions.',
        tuning_limitation='Seeds 0/1 were used to select DACER settings; report seeds 2/3/4 separately. Thresholds below describe observed effect size, not a statistical significance claim.')
    for metric in ('last5_success','final100_success'):
        result['iBOLT'][metric]=aggregate([r[metric] for r in rows]);result['SAC'][metric]=aggregate([r[metric] for r in sac])
        delta=[a[metric]-b[metric] for a,b in zip(rows,sac)];result['paired_gaps'][metric]=aggregate(delta)
        result.setdefault('additional_seeds_only',{})[metric]=dict(iBOLT=aggregate([r[metric] for r in rows[2:]]),
            SAC=aggregate([r[metric] for r in sac[2:]]),paired_gap=aggregate(delta[2:]))
    result['iBOLT']['final100_height']={name:dict(successes=sum(r['final100_height'][name]['final_successes'] for r in rows),
        episodes=sum(r['final100_height'][name]['episodes'] for r in rows)) for name in ('ground','low_air','air_over5cm')}
    result['SAC']['final100_height']=reference['final100_height']
    result['large_improvement_target']=dict(minimum_gap_percentage_points=10,
        requires='At least +10pp over SAC on both 5-seed last5 and final100 means, and positive mean final100 gap on additional seeds 2/3/4.')
    result['large_improvement_observed']=(result['paired_gaps']['last5_success']['mean']>=.1-1e-12 and
        result['paired_gaps']['final100_success']['mean']>=.1-1e-12 and
        result['additional_seeds_only']['final100_success']['paired_gap']['mean']>0)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    path=args.plan.resolve();folder=path.parent;plan=json.loads(path.read_text())
    profiles=plan.get('parameter_grid',[(h,l) for h in TARGETS for l in NOISE_SCALES])
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from fetch_rebuttal.campaign import environment,gpu_memory,memory_available_mib,process_live
    children={};stopping=False
    def stop(_sig,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def start(job,smoke=False):
        spec=folder/'specs'/(job['id']+'.json');atomic(spec,job)
        command=['taskset','-c',','.join(map(str,job['cpus'])),plan['python'],'-u','-m',
            'rebuttal_tools.fetch_dacer_worker','--spec',str(spec)]
        if smoke:command+=['--smoke']
        env=dict(environment(plan['source'],job['gpu']),JAX_COMPILATION_CACHE_DIR=plan['jax_cache'],MPLCONFIGDIR=str(folder/'mpl-cache'))
        with (folder/'logs'/(job['id']+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=plan['source'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        info=process_info(child.pid);assert info is not None
        job.update(status='initializing',pid=child.pid,start_ticks=info['start_ticks'],started_unix=time.time())
        children[child.pid]=child;print('Started '+job['id']+' GPU'+str(job['gpu']),flush=True);return child
    def preflight_gpu(gpu):
        result={};cases=profiles
        for index in range(gpu,len(cases),4):
            target,noise=cases[index];job=job_spec(plan,target,noise,100+index,'preflight')
            job.update(id='preflight-'+job['id'],gpu=gpu,cpus=plan['cpu_lanes'][gpu],output=str(folder/'preflights'/str(index)))
            child=start(job,True)
            if child.wait()!=0:raise RuntimeError('GPU preflight failed: '+job['id'])
            done=json.loads((Path(job['output'])/'dacer-verified-completion.json').read_text())
            assert done['updates']==4 and done['runtime']['backend']=='gpu' and done['alpha_update_direction_verified']
            job.update(status='completed',result=done);result[str(index)]=job
        return result
    def applications():
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
        values={}
        for row in csv.reader(io.StringIO(output)):
            try:values[int(row[0])]=int(row[1])
            except (ValueError,IndexError):pass
        return values
    try:
        plan.update(controller_pid=os.getpid(),status='validating_gpu');atomic(path,plan)
        if not plan.get('preflights_passed'):
            with ThreadPoolExecutor(max_workers=4) as pool:groups=list(pool.map(preflight_gpu,range(4)))
            plan['preflights']={k:v for group in groups for k,v in group.items()};assert len(plan['preflights'])==len(profiles)
            plan['preflights_passed']=True;atomic(folder/'gpu-preflight-verification.json',dict(passed=True,cases=plan['preflights']))
        plan['status']='running'
        while True:
            if stopping:raise InterruptedError('Stopped by request')
            for job in plan['jobs']:
                if job['status'] not in ('initializing','running'):continue
                out=Path(job['output']);marker=out/('verified-completion.json' if job['external'] else 'dacer-verified-completion.json')
                if marker.exists():
                    done=json.loads(marker.read_text());verify_saved_configuration(job)
                    assert done['updates']==995000 and done['runtime']['backend']=='gpu'
                    job.update(status='completed',result=done,finished_unix=done['finished_unix'])
                elif any((out/name).exists() for name in ('dacer-worker-failure.json','scale-worker-failure.json','failure.json')) or not process_live(job):
                    fails=[out/name for name in ('dacer-worker-failure.json','scale-worker-failure.json','failure.json') if (out/name).exists()]
                    reason=json.loads(fails[0].read_text()).get('error') if fails else 'Exited without verified completion'
                    job.update(status='failed',error=reason,finished_unix=time.time())
                elif (out/'progress.json').exists():
                    job['progress']=json.loads((out/'progress.json').read_text())
                    if job['progress'].get('updates',0)>0:job['status']='running'
            for pid,child in list(children.items()):
                if child.poll() is not None:child.wait();del children[pid]
            search=[j for j in plan['jobs'] if j['phase']=='search'];assert len(search)==2*len(profiles)
            if not plan.get('selection'):
                if any(j['status']=='failed' for j in search):plan['selection_blocked']='A search seed failed; no partial-grid selection.'
                elif all(j['status']=='completed' for j in search):
                    selection=choose_parameters(search,profiles);plan['selection']=selection;atomic(folder/'dacer-selection.json',selection)
                    c=selection['parameters']
                    if (c['target_entropy_per_dim'],c['noise_scale'])==(-.9,.1):
                        plan['jobs']+=plan['confirmation_reuse_candidates'];selection['confirmation_reused']=True
                    else:
                        plan['jobs']+=[job_spec(plan,c['target_entropy_per_dim'],c['noise_scale'],seed,'confirmation') for seed in (2,3,4)]
                        selection['confirmation_reused']=False
                    atomic(folder/'dacer-selection.json',selection)
            memory=gpu_memory();apps=applications();active=[j for j in plan['jobs'] if not j['external'] and j['status'] in ('initializing','running')]
            for job in plan['jobs']:
                if job['status']!='queued':continue
                if len(active)>=plan['max_new_active']:job['waiting_reason']='CPU concurrency admission';continue
                host_deficit=sum(max(0,plan['host_reserve_mib']-(process_info(j.get('pid',0)) or {}).get('rss_mib',0)) for j in active)
                if memory_available_mib()-host_deficit-plan['host_reserve_mib']<plan['host_headroom_mib']:
                    job['waiting_reason']='host memory reservation';continue
                choices=[]
                for gpu,(used,total) in memory.items():
                    own=[j for j in active if j['gpu']==gpu];deficit=sum(max(0,plan['gpu_reserve_mib']-apps.get(j['pid'],0)) for j in own)
                    projected=used+deficit+plan['gpu_reserve_mib']
                    if projected<=total-plan['gpu_headroom_mib'] and sum(j['status']=='initializing' for j in own)<plan['max_initializing_per_gpu']:
                        choices.append((projected/total,gpu))
                if not choices:job['waiting_reason']='GPU reservation / initialization admission';continue
                job['gpu']=min(choices)[1];job['cpus']=min(plan['cpu_lanes'],key=lambda lane:sum(j.get('cpus')==lane for j in active))
                start(job);job.pop('waiting_reason',None);active.append(job)
            plan.update(counts=dict(Counter(j['status'] for j in plan['jobs'])),checked_unix=time.time(),gpu_memory=memory,
                new_active=sum(not j['external'] and j['status'] in ('initializing','running') for j in plan['jobs']))
            atomic(path,plan)
            if not any(plan['counts'].get(s,0) for s in ('queued','initializing','running')):
                plan['status']='completed_with_failures' if plan['counts'].get('failed') else 'completed'
                if plan['status']=='completed':
                    report=compare_to_sac(plan['jobs'],json.loads(Path(plan['sac_reference']).read_text()),plan['selection'])
                    atomic(folder/'comparison.json',report);plan['large_improvement_observed']=report['large_improvement_observed']
                atomic(path,plan);return
            time.sleep(10)
    except BaseException as error:
        # Existing control PIDs belong to the original campaign, never this process tree.
        owned={j['pid'] for j in plan['jobs'] if not j['external'] and j['status'] in ('initializing','running') and process_live(j)}
        for pid,child in children.items():
            if child.poll() is None:owned.add(pid)
        for pid in owned:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        plan.update(status='stopped' if stopping else 'controller_error',error=repr(error));atomic(path,plan);raise


if __name__=='__main__':main()
