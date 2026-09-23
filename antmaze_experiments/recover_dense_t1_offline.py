"""Manual recovery of T1 jobs blocked before learning by W&B connectivity.

The caller archives the failed attempt and stops its supervisor first. Reuse
verified preflight results and the original frozen learning source, changing
only W&B transport. Never resume a partially trained job with this utility.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .controller import job
from .run import write
from .register_dense_t1 import CAMPAIGN


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job')
    p.add_argument('--gpu',type=int)
    a=p.parse_args();root=a.root
    m=json.loads((root/'manifest.json').read_text())
    assert m['campaign']==CAMPAIGN and m['shard']==1
    assert m['wandb_mode']=='offline'
    recovery=json.loads((root/'network-recovery.json').read_text())
    ops=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ops,text=True).strip()
    assert sha==recovery['recovery_source_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ops,text=True).strip()
    assert os.environ['WANDB_MODE']=='offline'
    for entry in m['jobs']:
        key=entry['id'];old=root/recovery['archive']/ 'runs'/key
        assert not (old/'progress.json').exists() and not (old/'result.json').exists()
        config=json.loads((old/'config.json').read_text())
        assert config['source_commit']==m['source_commit'] and config['temperature']==1
        assert config['reward_profile']=='dense' and config['noveld_enabled'] is False
        pre=root/'preflight'/key
        proof=json.loads((pre/'result.json').read_text())
        assert proof['completed'] and proof['source_commit']==m['source_commit']
        assert proof['steps']==8448 and proof['updates']==8 and proof['rnd_updates']==0
        assert proof['checkpoint']['readback_verified'] and proof['checkpoint']['environment_reward_verified']
        assert json.loads((pre/'optiq-profile-verification.json').read_text())['verified']
    if a.job:
        assert a.gpu==recovery['gpus'][a.job]
        return job(root,a.job,a.gpu,phases=('runs',))
    lock=(root/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'status.json').exists(), 'Do not restart this recovery automatically'
    live={};completed=[];failed=[]
    for entry in m['jobs']:
        key=entry['id'];gpu=recovery['gpus'][key]
        env=dict(os.environ,OPTIQ_SOURCE_DIR=str(ops),CAMPAIGN_GPU=str(gpu))
        command=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-direct-gmm',
            sys.executable,'-m','antmaze_experiments.recover_dense_t1_offline',
            '--root',str(root),'--job',key,'--gpu',str(gpu)]
        with (root/'logs'/f'{key}-offline-job.log').open('x') as log:
            live[key]=(subprocess.Popen(command,cwd=ops,env=env,stdout=log,stderr=subprocess.STDOUT),entry,gpu)
    while live:
        for key,(proc,entry,gpu) in list(live.items()):
            code=proc.poll()
            if code is None:continue
            del live[key]
            if code==0:completed.append(key)
            else:failed.append(dict(id=key,returncode=code,gpu=gpu))
        write(root/'status.json',dict(source_commit=m['source_commit'],recovery_source_commit=sha,
            controller_pid=os.getpid(),time=time.time(),pending=[],pending_held=bool(failed),
            running=[dict(**entry,gpu=gpu,pid=proc.pid) for proc,entry,gpu in live.values()],
            completed=completed,failed=failed,wandb_mode='offline'))
        if failed:write(root/'failure.json',dict(failed=failed,pending_held=True,time=time.time()))
        if live:time.sleep(2)
    if failed:return 1
    write(root/'result.json',dict(completed=True,source_commit=m['source_commit'],
        recovery_source_commit=sha,results={k:json.loads((root/'runs'/k/'result.json').read_text()) for k in completed}))
    return 0


if __name__=='__main__':
    sys.exit(main())
