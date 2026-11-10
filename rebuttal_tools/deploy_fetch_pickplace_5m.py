"""Deploy ten fixed, fresh SAC/iBOLT runs on otherwise free server4 GPUs."""
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
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True)
    p.add_argument('--root',type=Path,default=Path('/workspace/heechan-fetch-sigma'))
    p.add_argument('--code',type=Path,required=True);p.add_argument('--ibolt-reference',type=Path,required=True)
    p.add_argument('--sac-python',type=Path,required=True)
    p.add_argument('--reference-plan',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();reference=json.loads(a.reference_plan.read_text())
    assert reference['status']=='completed'
    original_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert not any(line.strip().isdigit() for line in original_processes.splitlines()), 'GPU jobs appeared; do not overlap existing runs'
    free=[int(line.strip()) for line in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(free)==4 and min(free)>28000
    assert shutil.disk_usage(root).free>30*1024**3
    source=root/('source-'+a.campaign);source.mkdir(exist_ok=False)
    with tarfile.open(root/'fetch-her-frozen-source.tgz') as t:t.extractall(source,filter='data')
    native={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file()}
    assert native==json.loads((a.reference_plan.parent/'native-source-sha256.json').read_text())
    with tarfile.open(a.code) as t:t.extractall(source,filter='data')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    for name in ('fetch_vector_env.py','fetch_vector_replay.py','check_fetch_vector.py'):
        rel=Path('rebuttal_tools')/name
        assert (source/rel).read_bytes()==(Path(reference['source'])/rel).read_bytes(),str(rel)
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\noutputs/\n*.log\n')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Freeze five-seed SAC iBOLT PickAndPlace 5M UTD1 comparison'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/a.campaign;folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','logs','runs','preflights','references','mpl-cache','jax-cache'):(folder/name).mkdir()
    cfg=json.loads(a.ibolt_reference.read_text())
    assert cfg['temperature']==.25 and cfg['dacer']['noise_scale']==2 and cfg['dacer']['target_entropy_per_dim']==-.9
    actor=cfg['alg']['actor'];assert (actor['log_std_min'],actor['log_std_max'],actor['initial_log_std'])==(-5,1,-1)
    refpath=folder/'references'/'ibolt-winning-config.json';shutil.copyfile(a.ibolt_reference,refpath)
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    manifestpath=folder/'source-sha256.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    physical={}
    for cpu in sorted(os.sched_getaffinity(0)):
        d=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=(d.joinpath('physical_package_id').read_text().strip(),d.joinpath('core_id').read_text().strip())
        physical.setdefault(key,[]).append(cpu)
    lanes=list(physical.values())[:50];assert len(lanes)==50
    # Each GPU has both methods; 50 physical CPU cores stay below the server quota.
    assignment=[(0,1),(2,3),(1,0),(3,2),(0,1)]
    def job(method,seed,index,gpu,smoke=False):
        ident=f'pick-and-place-{method}-vec256-utd0p125-5m-s{seed}'
        return dict(id=ident,method=method,seed=seed,source=str(source),source_manifest=str(manifestpath),
            python=str(a.sac_python) if method=='sac' else str(root/'venv/bin/python'),
            gpu=gpu,learner_cpus=lanes[index*5],worker_cpus=lanes[index*5+1:index*5+5],
            ibolt_reference_config=str(refpath),output=str(folder/('preflights' if smoke else 'runs')/ident),status='queued')
    jobs=[]
    for seed in range(5):
        for j,method in enumerate(('sac','ibolt')):jobs.append(job(method,seed,len(jobs),assignment[seed][j]))
    plan=dict(campaign=a.campaign,host=4,source=str(source),python=str(root/'venv/bin/python'),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        native_files_verified=len(native),source_manifest=str(manifestpath),cpu_lanes=lanes,
        original_gpu_processes=[],sac_environment_status=str(root/'sac-cu128-20261110-status.json'),jobs=jobs,preflights=[job('sac',100,0,0,True),job('ibolt',100,1,1,True)],
        created_unix=time.time(),status='prepared',config=dict(methods=['sac','ibolt'],seeds=[0,1,2,3,4],
            num_envs=256,batch_size=256,utd=.125,gradient_steps=32,warmup=12800,total_transitions=5000192,
            expected_updates=623424,replay_capacity=1000000,HER=True,normalize_observations=True,
            temperature=.25,log_std_min=-5,log_std_max=1,initial_log_std=-1,
            target_entropy_per_dim=-.9,noise_scale=2.,alpha_lr=.03,interval_updates=10000),
        stop_policy='Every fresh run stops at 5,000,192 transitions, the first vec256 boundary at or above 5M.',
        reporting='Five seed last5 and final100 means, seed SD and paired gaps; height-stratified approach/contact/lift/hold. No adaptive tuning or seed selection.')
    path=folder/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    program=a.campaign+'-host4'
    config=f'''[program:{program}]
command={root}/venv/bin/python -u -m rebuttal_tools.fetch_pickplace_5m_campaign --plan {path}
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
    for command in (['reread'],['add',program],['start',program]):
        subprocess.run(['sudo','-n','supervisorctl',*command],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(dict(program=program,plan=str(path),source_commit=plan['source_commit'],
        native_files_verified=len(native),jobs=10,preflights=2)),flush=True)


if __name__=='__main__':main()
