"""Committed four-GPU parameter screen; no learner changes or automatic retries."""
import argparse,ast,hashlib,json,os,subprocess,time
from pathlib import Path
from .campaign import read,write,occupied_gpus

BASE='17cfcc1ea82618e838fb692d254cbfceeefdfa9b'
CTL=['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
PYTHON='/home/heechan/.venv-optiq-gmm40/bin/python'


def unchanged_algorithm(source):
    def methods(code):
        cls=next(x for x in ast.parse(code).body if isinstance(x,ast.ClassDef) and x.name=='OptiQTRG')
        return {x.name:ast.dump(x,include_attributes=False) for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('_update','_sample')}
    path='gmm40/optiq_trg.py'
    old=subprocess.check_output(['git','show',BASE+':'+path],cwd=source,text=True)
    assert methods(old)==methods((source/path).read_text()),'Learner or sampler changed'
    prefix='analysis_tools/experiments/20260920_truncated_mll/optiq_dime/'
    checks={}
    for name in ['policy.py','semi_implicit.py','distillation.py','box_gaussian.py']:
        p=prefix+name;old=subprocess.check_output(['git','show',BASE+':'+p],cwd=source)
        assert old==(source/p).read_bytes(),p
        checks[p]=hashlib.sha256(old).hexdigest()
    return checks


def prepare(root,source):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    plan=read(source/'gmm40/mu90_screen_plan.json');sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if (root/'manifest.json').exists():
        m=read(root/'manifest.json');assert m['source_commit']==sha and m['plan']==plan;return
    jobs=[]
    for profile in plan['profiles']:
        name=plan['name']+'-'+profile['name']+'-s0'
        argv=['--method','optiq_trg','--name',name,'--seed','0']
        values={k:plan[k] for k in ['steps','batch','temperature','width','depth','eval_samples','trg_log_std_max','trg_initial_log_std','trg_teacher_std_floor']}
        values.update({k:profile[k] for k in ['n','m','mean_output_init_scale']})
        for k,v in values.items():argv+=['--'+k.replace('_','-'),str(v)]
        jobs.append(dict(name=name,method='optiq_trg',seed=0,steps=plan['steps'],gpu=profile['gpu'],profile=profile,args=argv))
    write(root/'manifest.json',dict(source=str(source),source_commit=sha,plan=plan,jobs=jobs,algorithm_unchanged=unchanged_algorithm(source),created=time.time()))
    write(root/'results/queue.json',dict(jobs=jobs))


def preflight(root,m):
    import gc,jax,numpy as np
    from .target import initialize_target
    from .optiq_trg import OptiQTRG
    assert jax.default_backend()=='gpu'
    target=initialize_target();checks=[];p=m['plan']
    for spec in p['profiles']:
        a=OptiQTRG(target,n=spec['n'],m=spec['m'],batch=p['batch'],
            log_std_max=p['trg_log_std_max'],initial_log_std=p['trg_initial_log_std'],
            teacher_std_floor=p['trg_teacher_std_floor'],mean_output_init_scale=spec['mean_output_init_scale'])
        assert a.actor.mean_output_init_scale==spec['mean_output_init_scale']
        info=a.advance(2);x,_,extra=a.evaluate_samples(128,937)
        assert a.updates==2 and np.isfinite(x).all() and np.max(np.abs(x))<=40
        assert np.isfinite(extra['mu_only']).all() and all(np.isfinite(v) for v in info.values())
        checks.append(dict(profile=spec,metrics=info));del a;gc.collect();jax.clear_caches()
    write(root/'preflight.json',dict(status='passed',source_commit=m['source_commit'],checks=checks,time=time.time()))
    print(json.dumps(checks),flush=True)


def launch(root,m):
    assert read(root/'preflight.json')['source_commit']==m['source_commit']
    assert read(root/'preflight.json')['status']=='passed'
    assert not occupied_gpus(),'Wait for free GPUs; never interrupt another experiment'
    d=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    for j in m['jobs']:
        assert not (d/(j['name']+'.conf')).exists()
        assert not (root/'results'/j['name']).exists()
    (root/'logs').mkdir(exist_ok=True);(root/'wandb').mkdir(exist_ok=True)
    for j in m['jobs']:
        env=dict(OPTIQ_SOURCE_DIR=m['source'],GMM40_REPO_ROOT=m['source'],GMM40_RESULTS_ROOT=str(root/'results'),GMM40_SOURCE_COMMIT=m['source_commit'],GMM40_CAMPAIGN=m['plan']['name'],GMM40_WANDB_DIR=str(root/'wandb'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
        command=['/home/heechan/OptiQ-ops/run-gpu.sh',str(j['gpu']),'--branch','v5-gmm40',PYTHON,'-u','-m','gmm40.campaign_job',*j['args']]
        conf='\n'.join([f"[program:{j['name']}]",'command='+' '.join(command),'directory='+m['source'],'autostart=true','autorestart=false','startsecs=3','startretries=0','stopasgroup=true','killasgroup=true','stopwaitsecs=30','redirect_stderr=true','stdout_logfile='+str(root/'logs'/(j['name']+'.log')),'stdout_logfile_maxbytes=10MB','stdout_logfile_backups=2','environment='+','.join(k+'="'+v+'"' for k,v in env.items()),''])
        (d/(j['name']+'.conf')).write_text(conf)
    write(root/'registration.json',dict(time=time.time(),source_commit=m['source_commit'],services=[j['name'] for j in m['jobs']],automatic_restarts=False))
    subprocess.run(CTL+['reread'],check=True)
    for j in m['jobs']:subprocess.run(CTL+['update',j['name']],check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','launch']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);source=Path(__file__).resolve().parents[1]
    os.environ.update(GMM40_REPO_ROOT=str(source),GMM40_RESULTS_ROOT=str(root/'results'))
    if a.action=='prepare':prepare(root,source)
    elif a.action=='preflight':preflight(root,read(root/'manifest.json'))
    else:launch(root,read(root/'manifest.json'))


if __name__=='__main__':main()
