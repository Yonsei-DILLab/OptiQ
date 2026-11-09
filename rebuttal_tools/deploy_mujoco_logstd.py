"""Prepare a pinned private source worktree and supervised MuJoCo campaign."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',type=int,choices=(1,2),required=True)
    parser.add_argument('--campaign',required=True);parser.add_argument('--revision',required=True)
    parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args()
    root=Path('/workspace/iBOLT')
    assert git(root,'branch','--show-current')=='heechan-rebuttal'
    original_head=git(root,'rev-parse','HEAD')
    folder=root/'outputs/mujoco-rebuttal'/args.campaign/f'host{args.host}'
    folder.mkdir(parents=True,exist_ok=False);source=folder/'source'
    reference='refs/rebuttal-runs/'+args.campaign
    subprocess.run(['git','-C',str(root),'fetch','--no-tags',str(args.bundle),
                    'refs/heads/heechan-rebuttal:'+reference],check=True)
    assert git(root,'rev-parse',reference)==args.revision
    branch='rebuttal-runs/'+args.campaign+f'-host{args.host}'
    subprocess.run(['git','-C',str(root),'worktree','add','-b',branch,str(source),args.revision],check=True)
    assert git(root,'rev-parse','HEAD')==original_head
    assert git(source,'rev-parse','HEAD')==args.revision
    assert 'anonymous' not in git(source,'branch','--show-current').lower()
    sys.path.insert(0,str(source))
    from rebuttal_tools.mujoco_logstd_campaign import make_plan
    from rebuttal_tools.mujoco_logstd_worker import atomic_json
    used=Counter()
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if 'python' not in (p/'comm').read_text():continue
            affinity=os.sched_getaffinity(int(p.name))
            if len(affinity)<=8:used.update(affinity)
        except (OSError,ProcessLookupError):pass
    allowed=os.sched_getaffinity(0);lanes=set()
    for cpu in allowed:
        siblings=set()
        for part in Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
            if '-' in part:
                lo,hi=map(int,part.split('-'));siblings.update(range(lo,hi+1))
            else:siblings.add(int(part))
        lanes.add(tuple(sorted(siblings&allowed)))
    lanes=sorted(lanes,key=lambda c:(sum(used[x] for x in c),c))[:32 if args.host==1 else 48]
    python='/workspace/iBOLT-runs/fetch-20261108-venv/bin/python'
    plan=make_plan(folder,source,python,args.host,args.revision,[list(c) for c in lanes])
    plan.update(campaign=args.campaign,primary_root_head_unchanged=original_head,
        source_worktree_branch=branch,bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest())
    atomic_json(folder/'plan.json',plan)
    username=pwd.getpwuid(os.getuid()).pw_name
    for suffix,extra in [('-preflight',' --preflight-only'),('','')]:
        program=args.campaign+f'-host{args.host}'+suffix
        config=f'''[program:{program}]
command={python} -u -m rebuttal_tools.mujoco_logstd_campaign --plan {folder}/plan.json{extra}
directory={source}
user={username}
autostart=false
autorestart=false
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={folder}/controller{suffix}.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
environment=PYTHONPATH="{source}",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"
'''
        subprocess.run(['sudo','-n','tee',f'/etc/supervisor/conf.d/{program}.conf'],input=config,text=True,
                       stdout=subprocess.DEVNULL,check=True)
        subprocess.run(['sudo','-n','supervisorctl','reread'],stdout=subprocess.DEVNULL,check=True)
        subprocess.run(['sudo','-n','supervisorctl','add',program],stdout=subprocess.DEVNULL,check=True)
    program=args.campaign+f'-host{args.host}-preflight'
    subprocess.run(['sudo','-n','supervisorctl','start',program],check=True)
    print(json.dumps(dict(host=args.host,prepared=True,program=program,folder=str(folder),
        jobs=len(plan['jobs']),source_commit=args.revision,gpus=dict(Counter(j['gpu'] for j in plan['jobs'])))),flush=True)


if __name__=='__main__':main()
