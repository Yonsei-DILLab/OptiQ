"""Deploy the DACER-only experiment separately from all running Fetch jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--root',type=Path,default=Path('/workspace/heechan-fetch-sigma'))
    parser.add_argument('--code',type=Path,required=True);parser.add_argument('--parent-plan',type=Path,required=True)
    parser.add_argument('--sac-reference',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();parent=json.loads(args.parent_plan.read_text());assert (root/'environment-ready').exists()
    source=root/('source-'+args.campaign);source.mkdir(exist_ok=False)
    with tarfile.open(root/'fetch-her-frozen-source.tgz') as tar:tar.extractall(source,filter='data')
    native={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    reference=root/'outputs/fetch-sigma-vec256-utd0125-20261109T183107Z-r2/native-source-sha256.json'
    assert native==json.loads(reference.read_text()),'Native source differs from the reference'
    with tarfile.open(args.code) as tar:tar.extractall(source,filter='data')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\n*.log\noutputs/\n')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Configure isolated iBOLT DACER-only PickAndPlace search'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/args.campaign;folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','logs','runs','preflights','mpl-cache','jax-cache'):(folder/name).mkdir()
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    manifestpath=folder/'source-sha256.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    shutil.copy2(args.sac_reference,folder/'sac-reference.json')
    controls=[j for j in parent['jobs'] if j['task']=='pick-and-place' and j['method']=='ibolt' and j['temperature']==.25]
    assert sorted(j['seed'] for j in controls)==list(range(5))
    reference_job=next(j for j in controls if j['seed']==0);refout=Path(reference_job['output'])
    def physical(cpu):
        base=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return (base.joinpath('physical_package_id').read_text().strip(),base.joinpath('core_id').read_text().strip())
    reserved={physical(cpu) for lane in parent['cpu_lanes'] for cpu in lane};representatives={}
    for cpu in sorted(os.sched_getaffinity(0)):
        key=physical(cpu)
        if key not in reserved:representatives.setdefault(key,cpu)
    lanes=[[cpu] for cpu in sorted(representatives.values())];assert len(lanes)>=4
    plan=dict(campaign=args.campaign,host=4,folder=str(folder),source=str(source),python=str(root/'venv/bin/python'),
        source_manifest=str(manifestpath),jax_cache=str(folder/'jax-cache'),cpu_lanes=lanes,
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        native_files_verified=len(native),reference_parent_plan=str(args.parent_plan),
        reference_learner_config=str(refout/'mujoco-config.json'),reference_run_config=str(refout/'config.json'),
        sac_reference=str(folder/'sac-reference.json'),preflights_passed=False,selection=None,jobs=[],
        created_unix=time.time(),status='prepared',max_new_active=min(20,len(lanes)),
        gpu_reserve_mib=2048,gpu_headroom_mib=4096,host_reserve_mib=4096,host_headroom_mib=16384,max_initializing_per_gpu=2,
        config=dict(task='pick-and-place',method='ibolt',temperature=.25,log_std_min=-5,log_std_max=1,initial_log_std=-1,
            num_envs=1,utd=1,batch_size=256,HER=True,normalize_observations=True,steps=1000000,warmup=5000,
            expected_updates=995000,unchanged_core_and_vendor_files=True),
        scope=dict(target_entropy_per_dim=[-.9,.5,.75,1.],noise_scale=[.1,.3,1.],search_seeds=[0,1],
            search_configurations=12,search_runs=24,reused_search_runs=2,new_search_runs=22,additional_seeds=[2,3,4]),
        success_target='Observed gains >=10 percentage points over SAC on both full-5-seed last5 and final100 means, plus positive final100 gap on additional seeds 2/3/4; not a statistical significance claim.',
        stop_policy='1M transitions per run. Existing experiments remain under their original controllers; cancelled temperature/scale searches remain cancelled.')
    sys.path.insert(0,str(source))
    from rebuttal_tools.fetch_dacer_campaign import job_spec
    from rebuttal_tools.fetch_dacer_worker import verify_saved_configuration
    def adopt(seed,phase):
        original=next(j for j in controls if j['seed']==seed);job=job_spec(plan,-.9,.1,seed,phase)
        job.update(external=True,external_original_id=original['id'],source=parent['source'],source_manifest=parent['source_manifest'],
            output=original['output'],pid=original.get('pid'),start_ticks=original.get('start_ticks'),
            gpu=original.get('gpu'),cpus=original.get('cpus'),status='running')
        verify_saved_configuration(job);return job
    for target in (.75,.5,1.,-.9):
        for noise in (.3,.1,1.):
            for seed in (0,1):
                plan['jobs'].append(adopt(seed,'search') if (target,noise)==(-.9,.1) else job_spec(plan,target,noise,seed))
    assert len(plan['jobs'])==24 and sum(j['external'] for j in plan['jobs'])==2
    plan['confirmation_reuse_candidates']=[adopt(seed,'confirmation') for seed in (2,3,4)]
    path=folder/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    program=args.campaign+'-host4'
    config=f'''[program:{program}]
command={root}/venv/bin/python -u -m rebuttal_tools.fetch_dacer_campaign --plan {path}
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
    print(json.dumps(dict(program=program,plan=str(path),source_commit=plan['source_commit'],native_files_verified=len(native),
        search_runs=24,new_runs=22,reused_runs=2,max_new_active=plan['max_new_active'])),flush=True)


if __name__=='__main__':main()
