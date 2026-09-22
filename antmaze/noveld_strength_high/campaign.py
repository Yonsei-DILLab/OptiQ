"""Two fresh 100k runs for coefficients 50 and 100; preflight precedes production."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from antmaze.evaluation import atomic_json

NAME='antmaze-v3-optiq-noveld-high-100k-s0-20260922'
PROFILE='dense-noveld-strength-100k'
COEFFICIENTS=(50.,100.)
PYTHON='/home/heechan/.venv-optiq-antmaze/bin/python'
OPS=Path('/home/heechan/OptiQ-ops')


def job_id(coefficient):return 'v3-optiq-c'+format(coefficient,'g').replace('.','p')+'-s0'


def source_sha():
    source=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    return source,sha


def verify(folder,coefficient,smoke=False):
    from antmaze.multimodal.dense_noveld_report import verify_run
    return verify_run(folder,allow_smoke=smoke,expected_source=source_sha()[1],
        expected_steps=100000,expected_profile=PROFILE,expected_coefficient=coefficient)


def worker(root,coefficient,preflight):
    job=job_id(coefficient)
    def call(folder,steps,*args):
        subprocess.run([sys.executable,'-u','-m','antmaze.multimodal.dense_noveld_run',
            '--task','v3','--method','optiq','--output',str(folder),'--steps',str(steps),
            '--noveld-coefficient',str(coefficient),'--campaign-name',NAME,'--run-name',job+'-100k',
            '--profile',PROFILE,*args],check=True)
    if preflight:
        if coefficient==COEFFICIENTS[0]:
            from .check_coefficient import check
            atomic_json(root/'coefficient-proof.json',check())
        first=root/'preflight'/job/'initial';second=root/'preflight'/job/'continued'
        call(first,272,'--smoke');verify(first,coefficient,True)
        call(second,280,'--smoke','--resume',str(first/'resume'/'step_0000000272'))
        proof=verify(second,coefficient,True)
        restored=json.loads((second/'resume-verification.json').read_text())
        assert restored['passed'] and restored['loaded_step']==272 and restored['loaded_updates']==16
        proof['resume_verification']=restored
        atomic_json(root/'proofs'/(job+'.json'),proof)
    else:
        folder=root/'runs'/job;call(folder,100000)
        atomic_json(folder/'verification.json',verify(folder,coefficient))


def match_initial(root,jobs,stage):
    records=[]
    for j in jobs:
        folder=root/'preflight'/j['id']/'initial' if stage=='preflight' else root/'runs'/j['id']
        c=json.loads((folder/'config.json').read_text())
        c['intrinsic'].pop('coefficient')
        c['native'].pop('output_root',None)
        records.append(dict(config=c,initial=json.loads((folder/'parameter-audit.json').read_text())['initial'],
            rnd=json.loads((folder/'intrinsic-audit.json').read_text())['initial']))
    assert all(r==records[0] for r in records),'Non-ablation settings or initial parameters differ'
    previous=Path('/home/heechan/optiq-experiments/antmaze-v3-optiq-noveld-strength-100k-s0-20260922')
    old_folder=previous/'preflight'/'v3-optiq-c0p1-s0'/'initial' if stage=='preflight' else previous/'runs'/'v3-optiq-c0p1-s0'
    old_config=json.loads((old_folder/'config.json').read_text())
    old_config['intrinsic'].pop('coefficient');old_config['native'].pop('output_root',None)
    old_record=dict(config=old_config,initial=json.loads((old_folder/'parameter-audit.json').read_text())['initial'],
        rnd=json.loads((old_folder/'intrinsic-audit.json').read_text())['initial'])
    comparable=json.loads(json.dumps(records[0]))
    for record in (old_record,comparable):
        record['config'].pop('source_commit');record['config'].pop('source_root')
    assert old_record==comparable,'New run differs from prior coefficient sweep beyond coefficient/source/output'
    atomic_json(root/f'{stage}-matched-initialization.json',dict(passed=True,jobs=[j['id'] for j in jobs],
        prior_sweep_reference=str(old_folder),prior_sweep_matched=True,reference=records[0]))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=float,choices=COEFFICIENTS);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();root=a.root
    if a.worker is not None:return worker(root,a.worker,a.preflight)
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'manifest.json').exists(),'Refuse duplicate launch'
    source,sha=source_sha()
    jobs=[dict(id=job_id(c),coefficient=c,method='optiq',task='v3',seed=0,steps=100000,gpu=g) for g,c in enumerate(COEFFICIENTS)]
    for name in ('logs','jobs','proofs','runs'):(root/name).mkdir(exist_ok=True)
    atomic_json(root/'manifest.json',dict(source=str(source),source_commit=sha,project='OptiQ/gmm-trg',
        group=NAME,profile=PROFILE,total=len(COEFFICIENTS),jobs=jobs,only_changed_hyperparameter='NovelD coefficient'))
    for stage in ('preflight','training'):
        for j in jobs:j.update(status='pending',stage=stage)
        live={};failed=False
        while True:
            for gpu,(proc,j,log) in list(live.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del live[gpu]
                proof=root/'proofs'/(j['id']+'.json') if stage=='preflight' else root/'runs'/j['id']/'verification.json'
                passed=code==0 and proof.exists() and json.loads(proof.read_text())['passed']
                j.update(status='completed' if passed else 'failed',exit_code=code,finished=time.time())
                failed|=not passed
                if not passed:atomic_json(root/'failure.json',dict(job=j,pending_held=True,live_preserved=True,automatic_restart=False))
                atomic_json(root/'jobs'/(j['id']+'.json'),j)
            if not failed:
                for j in jobs:
                    gpu=j['gpu']
                    if j['status']!='pending':continue
                    with (OPS/'locks'/f'gpu-{gpu}.lock').open('a') as probe:
                        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:continue
                        fcntl.flock(probe,fcntl.LOCK_UN)
                    cmd=[str(OPS/'run-gpu.sh'),str(gpu),'--branch','v5-direct-gmm',PYTHON,'-u','-m',__spec__.name,
                         '--root',str(root),'--worker',str(j['coefficient'])]
                    if stage=='preflight':cmd.append('--preflight')
                    env=os.environ.copy();env.update(OPTIQ_SOURCE_DIR=str(source),CAMPAIGN_GPU=str(gpu),PYTHONPATH=str(source))
                    log=(root/'logs'/f'{stage}-{j["id"]}.log').open('w')
                    proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                    j.update(status='running',pid=proc.pid,started=time.time(),command=cmd)
                    atomic_json(root/'jobs'/(j['id']+'.json'),j);live[gpu]=(proc,j,log)
            done=sum(j['status']=='completed' for j in jobs)
            atomic_json(root/'status.json',dict(stage=stage,phase='failed' if failed else 'running',updated=time.time(),
                source_commit=sha,completed=done,total=len(COEFFICIENTS),jobs=jobs))
            if not live and (failed or done==len(COEFFICIENTS)):
                if failed:raise RuntimeError(f'{stage} failed; no automatic restart')
                break
            time.sleep(2)
        try:
            match_initial(root,jobs,stage)
            assert json.loads((root/'coefficient-proof.json').read_text())['passed']
        except Exception as error:
            atomic_json(root/'failure.json',dict(stage=stage,type=type(error).__name__,message=str(error),
                pending_held=True,live_preserved=True,automatic_restart=False))
            raise
    atomic_json(root/'status.json',dict(phase='completed',stage=stage,source_commit=sha,completed=len(COEFFICIENTS),total=len(COEFFICIENTS),jobs=jobs))
    atomic_json(root/'result.json',dict(completed=True,source_commit=sha,jobs=jobs))


if __name__=='__main__':main()
