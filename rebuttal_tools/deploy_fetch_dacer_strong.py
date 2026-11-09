"""Independent strong-exploration DACER cohort on another available server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--root',type=Path,default=Path('/workspace/heechan-fetch-dacer'))
    parser.add_argument('--code',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();refs=root/'references';source=root/('source-'+args.campaign);source.mkdir(exist_ok=False)
    with tarfile.open(root/'fetch-her-frozen-source.tgz') as tar:tar.extractall(source,filter='data')
    native={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    assert native==json.loads((refs/'native-source-sha256.json').read_text())
    with tarfile.open(args.code) as tar:tar.extractall(source,filter='data')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\n*.log\noutputs/\n')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Add separate DACER exploration-first cohort without changing native learners'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/args.campaign;folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','logs','runs','preflights','mpl-cache','jax-cache'):(folder/name).mkdir()
    manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    manifestpath=folder/'source-sha256.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    app_output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    original=[];reserved=set()
    def physical(cpu):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return (p.joinpath('physical_package_id').read_text().strip(),p.joinpath('core_id').read_text().strip())
    for line in app_output.splitlines():
        if not line.strip().isdigit():continue
        pid=int(line.strip())
        try:
            cpus=sorted(os.sched_getaffinity(pid));fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        except (OSError,ProcessLookupError):continue
        original.append(dict(pid=pid,start_ticks=int(fields[19]),cpus=cpus))
        if len(cpus)<=8:reserved|={physical(cpu) for cpu in cpus}
    available={}
    for cpu in sorted(os.sched_getaffinity(0)):
        key=physical(cpu)
        if key not in reserved:available.setdefault(key,[]).append(cpu)
    lanes=list(available.values())[:8];assert len(lanes)==8
    plan=dict(campaign=args.campaign,host=3,folder=str(folder),source=str(source),python=str(root/'venv/bin/python'),
        source_manifest=str(manifestpath),jax_cache=str(folder/'jax-cache'),cpu_lanes=lanes,
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),native_files_verified=len(native),
        reference_learner_config=str(refs/'reference-learner.json'),reference_run_config=str(refs/'reference-run.json'),
        reference_parent_plan='Host4 frozen configuration copied into references; original jobs remain on host4.',
        sac_reference=str(refs/'sac-reference.json'),parameter_grid=[[-.9,2.],[-.9,3.],[.5,2.],[.5,3.]],
        preflights_passed=False,selection=None,jobs=[],confirmation_reuse_candidates=[],original_gpu_processes=original,
        created_unix=time.time(),status='waiting_environment',max_new_active=8,
        gpu_reserve_mib=2048,gpu_headroom_mib=4096,host_reserve_mib=4096,host_headroom_mib=16384,max_initializing_per_gpu=2,
        config=dict(task='pick-and-place',method='ibolt',temperature=.25,log_std_min=-5,log_std_max=1,initial_log_std=-1,
            num_envs=1,utd=1,batch_size=256,HER=True,normalize_observations=True,steps=1000000,warmup=5000,expected_updates=995000),
        rationale='Test stronger initial random exploration with DACER alpha naturally decaying later, rather than only increasing its entropy floor.',
        global_selection_rule='Compare cohort leaders using only seeds0/1 pre-final last5, high-air tie-break and AUC. Additional seeds2/3/4 remain confirmation, not a selection criterion.',
        scope=dict(search_configurations=4,search_runs=8,new_search_runs=8,search_seeds=[0,1],additional_seeds=[2,3,4]))
    sys.path.insert(0,str(source));from rebuttal_tools.fetch_dacer_campaign import job_spec
    for target,noise in plan['parameter_grid']:
        for seed in (0,1):plan['jobs'].append(job_spec(plan,target,noise,seed))
    path=folder/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    wrapper=folder/'wait-and-run.sh';wrapper.write_text(f'''#!/bin/bash
set -eu
while [ ! -f {root}/environment-ready ]; do sleep 5; done
exec {root}/venv/bin/python -u -m rebuttal_tools.fetch_dacer_campaign --plan {path}
''')
    program=args.campaign+'-host3'
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
redirect_stderr=true
stdout_logfile={folder}/controller.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
environment=PYTHONPATH="{source}",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"
'''
    subprocess.run(['sudo','-n','tee','/etc/supervisor/conf.d/'+program+'.conf'],input=config,text=True,stdout=subprocess.DEVNULL,check=True)
    for command in (['reread'],['add',program],['start',program]):subprocess.run(['sudo','-n','supervisorctl',*command],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(dict(program=program,source_commit=plan['source_commit'],plan=str(path),jobs=8,native_files_verified=len(native))),flush=True)


if __name__=='__main__':main()
