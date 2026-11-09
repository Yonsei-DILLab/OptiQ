"""Freeze the SAC-matched Fetch rerun separately from existing experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--root',type=Path,default=Path('/workspace/heechan-fetch-sigma'))
    parser.add_argument('--code',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();source=root/('source-'+args.campaign)
    assert (root/'environment-ready').exists()
    source.mkdir(exist_ok=False)
    with tarfile.open(root/'fetch-her-frozen-source.tgz') as tar:tar.extractall(source,filter='data')
    native={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    reference=root/'outputs/fetch-sigma-vec256-utd0125-20261109T183107Z-r2/native-source-sha256.json'
    assert native==json.loads(reference.read_text()),'Native Fetch bytes differ from existing reference'
    with tarfile.open(args.code) as tar:tar.extractall(source,filter='data')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\n*.log\noutputs/\n')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Isolate wide-scale Fetch UTD1 benchmark rerun and staged temperature selection'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/args.campaign;folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','logs','runs','preflights','mpl-cache','jax-cache'):(folder/name).mkdir()
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    manifestpath=folder/'source-sha256.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    allowed=os.sched_getaffinity(0);lanes=set()
    for cpu in allowed:
        # A single hardware thread per physical core bounds each JAX worker's pools.
        physical=(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').read_text().strip(),
            Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text().strip())
        lanes.add((physical,cpu))
    representatives={}
    for physical,cpu in sorted(lanes,key=lambda x:x[1]):representatives.setdefault(physical,cpu)
    cpu_lanes=[[cpu] for cpu in sorted(representatives.values())];assert len(cpu_lanes)>=40
    import sys
    sys.path.insert(0,str(source))
    from rebuttal_tools.fetch_wide_utd1_campaign import job_spec,TEMPERATURES,TASKS
    plan=dict(campaign=args.campaign,host=4,folder=str(folder),source=str(source),python=str(root/'venv/bin/python'),
        source_manifest=str(manifestpath),jax_cache=str(folder/'jax-cache'),cpu_lanes=cpu_lanes[:44],
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        native_reference='Frozen HER Fetch source, parent 6eff3ddee16a6ea96fe8093ae91f96c085c22abd',
        native_files_verified=len(native),selections={},preflights_passed=False,jobs=[],created_unix=time.time(),status='prepared',
        gpu_reserve_mib=2048,gpu_headroom_mib=4096,host_reserve_mib=4096,host_headroom_mib=16384,max_initializing_per_gpu=2,
        config=dict(log_std_min=-5,log_std_max=1,initial_log_std=-1,num_envs=1,utd=1,batch_size=256,
            steps=1000000,warmup=5000,expected_updates=995000,HER=True,normalize_observations=True,
            lr_actor=.0003,lr_critic=.0003,actor_widths=[256,256],critic_widths=[256,256],gamma=.99,tau=.005,
            eval_every=5000,eval_episodes=10,final_episodes=100,checkpoint_every=100000),
        scope=dict(fixed_temperatures={'pick-and-place':.25,'push':.05},search_temperatures=list(TEMPERATURES),
            search_tasks=['slide','reach'],search_seeds=[0,1],selected_additional_seeds=[2,3,4],
            initial_runs=38,total_runs_including_search=44,selected_final_runs=20),
        stop_policy='Stop each run at 1M transitions. No extension of the separate vector sigma campaign.')
    groups={task:[] for task in TASKS}
    for task,temp in (('pick-and-place',.25),('push',.05)):
        groups[task]=[job_spec(plan,task,seed,temp,'fixed-5seed') for seed in range(5)]
    for task in ('slide','reach'):
        groups[task]=[job_spec(plan,task,seed,temp,'temperature-sweep') for seed in (0,1) for temp in TEMPERATURES]
    for index in range(14):
        for task in TASKS:
            if index<len(groups[task]):plan['jobs'].append(groups[task][index])
    assert len(plan['jobs'])==38 and len({j['output'] for j in plan['jobs']})==38
    planpath=folder/'plan.json';planpath.write_text(json.dumps(plan,indent=2)+'\n')
    program=args.campaign+'-host4'
    config=f'''[program:{program}]
command={root}/venv/bin/python -u -m rebuttal_tools.fetch_wide_utd1_campaign --plan {planpath}
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
    print(json.dumps(dict(program=program,plan=str(planpath),jobs=38,eventual_runs=44,
        native_files_verified=len(native),source_commit=plan['source_commit'])),flush=True)


if __name__=='__main__':main()
