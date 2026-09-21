"""Replicate the passing NM64 seed0 configuration on new training seeds."""
import os,subprocess,time
from pathlib import Path
from .campaign import prepare,read,write,occupied_gpus
from .mu90_launch import CTL,PYTHON,unchanged_algorithm
SOURCE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/heechan/optiq-experiments/gmm40-mu90-replication-nm64-20260921')
NAME=ROOT.name

def main():
    unchanged_algorithm(SOURCE)
    m=prepare(ROOT,SOURCE/'gmm40/mu90_replication_plan.json')
    assert m['plan']['optiq_n']==m['plan']['optiq_m']==64
    assert 2 not in occupied_gpus(),'Wait for the completed seed0 GPU slot'
    for d in ('logs','wandb'):(ROOT/d).mkdir(exist_ok=True)
    env=dict(os.environ,OPTIQ_SOURCE_DIR=str(SOURCE),GMM40_REPO_ROOT=str(SOURCE),GMM40_RESULTS_ROOT=str(ROOT/'results'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    subprocess.run(['/home/heechan/OptiQ-ops/run-gpu.sh','2','--branch','v5-gmm40',PYTHON,'-B','-u','-m','gmm40.campaign','preflight','--root',str(ROOT)],env=env,cwd=SOURCE,check=True)
    assert read(ROOT/'preflight.json')['status']=='passed'
    conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(NAME+'.conf');assert not conf.exists()
    conf.write_text('\n'.join([f'[program:{NAME}]',f'command={PYTHON} -B -u -m gmm40.campaign run --root {ROOT}',f'directory={SOURCE}','autostart=true','autorestart=false','startsecs=3','startretries=0','stopasgroup=true','killasgroup=true','stopwaitsecs=30','redirect_stderr=true',f'stdout_logfile={ROOT}/controller.log','stdout_logfile_maxbytes=10MB','environment=PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2"','']))
    write(ROOT/'registration.json',dict(service=NAME,source_commit=m['source_commit'],source=str(SOURCE),created=time.time(),automatic_restarts=False))
    subprocess.run(CTL+['reread'],check=True);subprocess.run(CTL+['update',NAME],check=True)
if __name__=='__main__':main()
