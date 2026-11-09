"""Prepare the six-job Fetch scale comparison on an otherwise separate host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--root',type=Path,default=Path('/workspace/heechan-fetch-sigma'))
    parser.add_argument('--code',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();source=root/'source-r2'
    source.mkdir(exist_ok=False)
    with tarfile.open(root/'fetch-her-frozen-source.tgz') as tar:tar.extractall(source,filter='data')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    assert subprocess.check_output(['git','-C',str(source),'branch','--show-current'],text=True).strip()=='heechan-rebuttal'
    native={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    with tarfile.open(args.code) as tar:tar.extractall(source,filter='data')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\n*.log\noutputs/\n')
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Add isolated vector collector and three-condition Fetch sigma experiment'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/args.campaign;folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','logs','runs','preflights','mpl-cache','jax-cache'):(folder/name).mkdir()
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts and f.suffix not in ('.log','.pyc')}
    manifestpath=folder/'source-sha256.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    allowed=os.sched_getaffinity(0);lanes=set()
    for cpu in allowed:
        siblings=set()
        for part in Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
            if '-' in part:
                lo,hi=map(int,part.split('-'));siblings.update(range(lo,hi+1))
            else:siblings.add(int(part))
        lanes.add(tuple(sorted(siblings&allowed)))
    lanes=sorted(lanes);assert len(lanes)>=30
    jobs=[];preflights=[]
    for v,variant in enumerate(('original','narrow','wide')):
        def make(seed,index,smoke=False):
            ident=f'pick-and-place-ibolt-{variant}-vec256-utd0p125-T0.25-s{seed}'
            return dict(id=ident,variant=variant,seed=seed,source=str(source),source_manifest=str(manifestpath),
                gpu=(v if smoke else (0,1,2,3,0,1)[index]),learner_cpus=list(lanes[index*5]),
                worker_cpus=[list(x) for x in lanes[index*5+1:index*5+5]],
                output=str(folder/('preflights' if smoke else 'runs')/ident),status='queued')
        preflights.append(make(100,v,True))
        for seed in (0,1):jobs.append(make(seed,v*2+seed))
    plan=dict(campaign=args.campaign,host=4,source=str(source),python=str(root/'venv/bin/python'),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        native_reference='Existing HER experiment host1 frozen source; parent reference 6eff3ddee16a6ea96fe8093ae91f96c085c22abd',
        jobs=jobs,preflights=preflights,created_unix=time.time(),status='waiting_environment',
        config=dict(num_envs=256,batch_size=256,utd=.125,gradient_steps=32,warmup=12800,total_transitions=1000192,expected_updates=123424))
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    wrapper=folder/'wait-and-run.sh'
    wrapper.write_text(f'''#!/bin/bash
set -eu
while [ ! -f {root}/environment-ready ]; do sleep 5; done
exec {root}/venv/bin/python -u -m rebuttal_tools.fetch_sigma_vector_campaign --plan {folder}/plan.json
''')
    program=args.campaign+'-host4'
    config=f'''[program:{program}]
command=/bin/bash {wrapper}
directory={source}
user=heechan
autostart=false
autorestart=false
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={folder}/controller.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
environment=PYTHONPATH="{source}",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"
'''
    subprocess.run(['sudo','-n','tee',f'/etc/supervisor/conf.d/{program}.conf'],input=config,text=True,stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sudo','-n','supervisorctl','reread'],stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sudo','-n','supervisorctl','add',program],stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sudo','-n','supervisorctl','start',program],check=True)
    print(json.dumps(dict(program=program,plan=str(folder/'plan.json'),jobs=6,preflights=3,native_files_verified=len(native),source_commit=plan['source_commit'])),flush=True)


if __name__=='__main__':main()
