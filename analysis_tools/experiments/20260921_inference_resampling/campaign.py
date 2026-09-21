"""Register evaluation-only jobs with immutable code and checkpoint provenance."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path('/home/heechan/optiq-experiments/trg-inference-resampling-20260921')
PYTHON='/home/heechan/.venv-optiq-mujoco/bin/python'
SOURCE=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','register','worker'])
    p.add_argument('--gpu',type=int);a=p.parse_args()
    manifest_path=ROOT/'manifest.json'
    if a.action=='prepare':
        assert not manifest_path.exists()
        jobs=[]
        antroot=Path('/home/heechan/optiq-experiments/trg-dacer-priority-20260921/outputs')
        for p in sorted(antroot.glob('ant-trg-dacer-T0.25-b1-s*/config.json')):
            jobs.append(dict(task='ant',config=str(p),
                actor=str(next(p.parent.glob('checkpoints/*/actor_state_1000000.msgpack'))),
                critic=str(next(p.parent.glob('checkpoints/*/critic_state_1000000.msgpack')))))
        for p in sorted((ROOT/'inputs').glob('*/config.json')):
            jobs.append(dict(task='halfcheetah',config=str(p),
                actor=str(p.parent/'actor_state_1000000.msgpack'),critic=str(p.parent/'critic_state_1000000.msgpack'),
                original_wandb='https://wandb.ai/OptiQ/gmm-trg/runs/'+p.parent.name))
        assert len(jobs)==9
        for job in jobs:
            c=json.loads(Path(job['config']).read_text())
            assert c['total_steps']==1000000 and c['dacer']['enabled'] and c['alg']['actor']['temperature']==.25
            assert c['alg']['actor']['density_beta']==1
            job.update(seed=c['seed'],name=job['task']+'-s'+str(c['seed']),
                source='/home/heechan/OptiQ-ops/sources/'+c['runtime']['git_commit'],
                training_commit=c['runtime']['git_commit'],checkpoint_steps=1000000,
                input_sha256={k:sha(Path(job[k])) for k in ['config','actor','critic']})
            assert Path(job['source']).exists()
        jobs.sort(key=lambda x:(x['task'],x['seed']))
        sha1=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
        manifest=dict(jobs=jobs,output=str(ROOT/'results'),evaluation_commit=sha1,
            episodes=20,seed_base=19210921,candidates=64,density_reference_samples=256,
            temperature=.25,critic='mean of two current trained critics (training source_q_eval)',
            proposal='mu only, fresh Gaussian latent; no conditional sigma or DACER noise',
            kde='independent 256 mu samples; diagonal Scott bandwidth max(std*n^(-1/(D+4)),.001); box-normalized Gaussian kernels',
            scope='post-hoc final-1M checkpoint comparison; not last-100k training average',
            modes=dict(mu_one='existing mu-only action',mu_q64='categorical softmax(Q/.25)',
                       mu_kde_is64='categorical softmax(Q/.25-log KDE_mu); approximate importance resampling'))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps([{k:j[k] for k in ['name','training_commit']} for j in jobs]))
    elif a.action=='worker':
        index=[0,1,3].index(a.gpu)
        jobs=json.loads(manifest_path.read_text())['jobs']
        for i in range(index,len(jobs),3):
            subprocess.run([PYTHON,str(Path(__file__).with_name('evaluate.py')),
                '--manifest',str(manifest_path),'--index',str(i)],check=True)
        print('Evaluation worker complete',flush=True)
    else:
        # Each worker holds an existing OptiQ GPU lock; do not interrupt any training.
        conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
        for gpu in [0,1,3]:
            name='trg-inference-resampling-20260921-gpu'+str(gpu)
            dst=conf/(name+'.conf');assert not dst.exists()
            dst.write_text(f'''[program:{name}]
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
        ctl=['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
        subprocess.run(ctl+['reread'],check=True)
        for gpu in [0,1,3]:
            name='trg-inference-resampling-20260921-gpu'+str(gpu)
            subprocess.run(ctl+['update',name],check=True)
            subprocess.run(ctl+['start',name],check=True)

if __name__=='__main__':main()
