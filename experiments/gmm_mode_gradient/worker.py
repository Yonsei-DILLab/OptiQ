"""Shared atomic claims: both Slurm QoS pools work on the same independent tasks."""
import os,sys,time,json,subprocess,socket
from pathlib import Path
from experiments.gmm_gradient_interference import bootstrap
from .run import write,verify
ROOT=bootstrap.ROOT;OUT=ROOT/'runtime';PLAN=json.loads((Path(__file__).parent/'plan.json').read_text())
def main():
    source=verify();assert json.loads((OUT/'VALIDATION.json').read_text())['passed']
    from .core import initialize,engine,jax,np
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    for method in PLAN['methods']:
        state,key=initialize(0);state,key,metrics=engine(16,16,method)['step'](state,key)
        assert np.isfinite(np.asarray(metrics)).all()
    write(OUT/'logs'/('GPU_VALIDATED_'+os.environ.get('SLURM_JOB_ID','local')+'.json'),dict(passed=True,devices=[str(x) for x in jax.devices()],commit=source['commit']))
    jobs=[(n,m,s,method) for s in PLAN['seeds'] for n,m in PLAN['sizes'] for method in PLAN['methods']]
    for n,m,seed,method in jobs:
        name=f'{method}_N{n}_M{m}_s{seed}';dest=OUT/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'COMPLETE.json').exists() or (dest/'FAILED.json').exists():continue
        claim=dest/'claim'
        try:claim.mkdir()
        except FileExistsError:continue
        cmd=[sys.executable,'-m','experiments.gmm_mode_gradient.run','--root',str(OUT),'--n',str(n),'--m',str(m),
             '--seed',str(seed),'--method',method,'--until',str(PLAN['updates'])]
        record=dict(name=name,commit=source['commit'],job_id=os.environ.get('SLURM_JOB_ID'),array_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                    hostname=socket.gethostname(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started=time.time(),command=cmd)
        write(dest/'LAUNCH.json',record)
        with (dest/'stdout.log').open('a') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(dest/'EXIT.json',dict(**record,exit_code=result.returncode,finished=time.time()))
        claim.rmdir()
        if result.returncode and not (dest/'FAILED.json').exists():
            write(dest/'FAILED.json',dict(error='process exit',code=result.returncode,time=time.time()))
        if result.returncode:print('FAILED',name,result.returncode,flush=True)
        else:print('FINISHED',name,flush=True)
if __name__=='__main__':main()
