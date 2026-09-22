"""Register inference-only workers after all learning has been stopped by the user."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

NAME='antmaze-noveld10-checkpoint-audit-20260922'
ROOT=Path('/home/heechan/optiq-experiments')/NAME
OPS=Path('/home/heechan/OptiQ-ops')
PYTHON='/home/heechan/.venv-optiq-antmaze/bin/python'

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--shard',type=int,choices=(0,1),required=True);p.add_argument('--gpu',type=int)
    a=p.parse_args();source=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    tasks=[('v1','optiq',300000),('v1','sac',200000),('v1','mfpo',300000),('v3','optiq',100000)] if a.shard==0 else [('v2','optiq',100000),('v2','sac',100000)]
    jobs=[]
    for task,method,step in tasks:
        for coefficient in (.01,10.):
            campaign='antmaze-dense-noveld-1m-s0-20260922' if coefficient==.01 else 'antmaze-dense-noveld10-1m-s0-20260922'
            run=Path('/home/heechan/optiq-experiments')/campaign/'runs'/f'{task}-{method}-s0'
            if task=='v3' and coefficient==10:
                run=Path('/home/heechan/optiq-experiments/antmaze-v3-optiq-noveld-strength-100k-s0-20260922/runs/v3-optiq-c10-s0')
            jobs.append(dict(id=f'{task}-{method}-c{coefficient:g}-{step}',run=str(run),step=step,coefficient=coefficient,task=task,method=method))
    if a.gpu is not None:
        for j in jobs[a.gpu::4]:
            out=ROOT/'runs'/j['id']
            subprocess.run([PYTHON,'-u',str(source/'antmaze/noveld10_checkpoint_audit/rollout.py'),
                '--run',j['run'],'--step',str(j['step']),'--output',str(out),'--diagnostic-source-sha',sha],check=True)
        return
    assert not ROOT.exists(),'Refuse duplicate audit registration'
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    for gpu in range(4):
        with (OPS/'locks'/f'gpu-{gpu}.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
    ROOT.mkdir();(ROOT/'logs').mkdir();(ROOT/'runs').mkdir()
    services=[]
    for gpu in range(4):
        name=NAME+f'-g{gpu}';conf=OPS/'supervisor/jobs'/f'{name}.conf';assert not conf.exists()
        command=[str(OPS/'run-gpu.sh'),str(gpu),'--branch','v5-direct-gmm',PYTHON,'-u',str(Path(__file__).resolve()),'--shard',str(a.shard),'--gpu',str(gpu)]
        conf.write_text(f'''[program:{name}]
command={' '.join(command)}
directory={source}
environment=OPTIQ_SOURCE_DIR="{source}",PYTHONPATH="{source}",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",XLA_PYTHON_CLIENT_PREALLOCATE="false"
autostart=false
autorestart=false
startsecs=5
startretries=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={ROOT}/logs/gpu-{gpu}.log
stdout_logfile_maxbytes=0
''');services.append(name)
    (ROOT/'manifest.json').write_text(json.dumps(dict(inference_only=True,source_commit=sha,shard=a.shard,jobs=jobs,services=services),indent=2)+'\n')
    ctl=['/usr/local/bin/supervisorctl','-c',str(OPS/'supervisor/supervisord.conf')]
    for args in (['reread'],['update',*services],['start',*services]):subprocess.run(ctl+args,check=True)
    print(json.dumps(dict(registered=True,inference_only=True,jobs=jobs,source_commit=sha)))

if __name__=='__main__':main()
