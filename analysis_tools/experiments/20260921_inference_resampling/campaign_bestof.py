"""User-requested argmax-Q evaluation, paired with the completed three-way study."""
from pathlib import Path
import argparse,json,subprocess

ROOT=Path('/home/heechan/optiq-experiments/trg-inference-bestof64-20260921')
BASE=Path('/home/heechan/optiq-experiments/trg-inference-resampling-20260921')
SOURCE=Path(__file__).resolve().parents[3]
PYTHON='/home/heechan/.venv-optiq-mujoco/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','register','worker']);p.add_argument('--gpu',type=int)
    a=p.parse_args();mp=ROOT/'manifest.json'
    if a.action=='prepare':
        assert not mp.exists();ROOT.mkdir(exist_ok=True)
        m=json.loads((BASE/'manifest.json').read_text())
        for j in m['jobs']:
            old=json.loads((BASE/'results'/(j['name']+'.json')).read_text())
            assert old['complete'] and old['checkpoint_state_unchanged']
        m.update(base_manifest=str(BASE/'manifest.json'),base_evaluation_commit=m['evaluation_commit'],
                 evaluation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
                 output=str(ROOT/'results'),evaluation_modes=['mu_best64'])
        m['modes']={'mu_best64':'64 fresh mu candidates; choose argmax current twin-critic mean Q; no sigma or DACER noise'}
        mp.write_text(json.dumps(m,indent=2)+'\n')
        print('Prepared same9 checkpoints and same20 reset/policy seeds for best-of64')
    elif a.action=='worker':
        for i in range([0,1,3].index(a.gpu),9,3):
            subprocess.run([PYTHON,str(Path(__file__).with_name('evaluate.py')),'--manifest',str(mp),'--index',str(i)],check=True)
        print('Best-of64 worker complete',flush=True)
    else:
        ctl=['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
        for gpu in [0,1,3]:
            name='trg-inference-bestof64-20260921-gpu'+str(gpu)
            p=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf');assert not p.exists()
            p.write_text(f'''[program:{name}]
command=/home/heechan/OptiQ-ops/run-gpu.sh {gpu} --branch v5 {PYTHON} {Path(__file__).resolve()} worker --gpu {gpu}
directory={SOURCE}
autostart=false
autorestart=false
startsecs=2
startretries=0
redirect_stderr=true
stdout_logfile={ROOT}/{name}.log
stdout_logfile_maxbytes=0
environment=PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",XLA_PYTHON_CLIENT_PREALLOCATE="false"
''')
        subprocess.run(ctl+['reread'],check=True)
        for gpu in [0,1,3]:
            name='trg-inference-bestof64-20260921-gpu'+str(gpu)
            subprocess.run(ctl+['update',name],check=True);subprocess.run(ctl+['start',name],check=True)

if __name__=='__main__':main()
