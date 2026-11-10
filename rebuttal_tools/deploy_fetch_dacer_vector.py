"""Two-seed H1.8/L2 diagnostic, preserving the frozen vec256 UTD1/8 learner."""
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
    p.add_argument('--code',type=Path,required=True);p.add_argument('--reference-plan',type=Path,required=True)
    p.add_argument('--reserve-plan',type=Path,action='append',default=[]);a=p.parse_args()
    root=a.root.resolve();source=root/('source-'+a.campaign);source.mkdir(exist_ok=False)
    with tarfile.open(root/'fetch-her-frozen-source.tgz') as t:t.extractall(source,filter='data')
    native={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in source.rglob('*') if f.is_file()}
    with tarfile.open(a.code) as t:t.extractall(source,filter='data')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    reference=json.loads(a.reference_plan.read_text());assert reference['status']=='completed'
    for name in ('fetch_sigma_vector_worker.py','fetch_vector_env.py','fetch_vector_replay.py'):
        rel=Path('rebuttal_tools')/name
        assert (source/rel).read_bytes()==(Path(reference['source'])/rel).read_bytes(),str(rel)
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\n*.log\noutputs/\n')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Add isolated DACER H1.8 L2 vector diagnostic'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/a.campaign;folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','logs','runs','preflights','references','mpl-cache','jax-cache'):(folder/name).mkdir()
    controls=sorted([j for j in reference['jobs'] if j['variant']=='wide'],key=lambda j:j['seed'])
    assert [j['seed'] for j in controls]==[0,1]
    for filename,dest in [('mujoco-config.json','reference-learner.json'),('config.json','reference-run.json')]:
        shutil.copyfile(Path(controls[0]['output'])/filename,folder/'references'/dest)
    control_rows=[]
    for job in controls:
        out=Path(job['output']);done=json.loads((out/'completed.json').read_text());assert done['steps']==1000192 and done['updates']==123424
        cfg=json.loads((out/'mujoco-config.json').read_text());assert cfg['dacer']['noise_scale']==.1 and cfg['dacer']['target_entropy_per_dim']==-.9
        es={r['steps']:r for r in map(json.loads,(out/'evaluations.jsonl').read_text().splitlines())}
        hs={r['steps']:r for r in map(json.loads,(out/'height-evaluations.jsonl').read_text().splitlines())}
        control_rows.append(dict(seed=job['seed'],output=job['output'],completed=done,
            evaluations=[es[t] for t in sorted(es)],final_height=hs[1000192]))
    (folder/'references/vector-control.json').write_text(json.dumps(dict(source_commit=reference['source_commit'],seeds=control_rows),indent=2)+'\n')
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    manifestpath=folder/'source-sha256.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    def physical(cpu):
        d=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return (d.joinpath('physical_package_id').read_text().strip(),d.joinpath('core_id').read_text().strip())
    reserved=set();reservations=[]
    for path in a.reserve_plan:
        old=json.loads(path.read_text())
        if old['status'] in ('completed','completed_with_failures','stopped','controller_error'):continue
        cpus={c for lane in old.get('cpu_lanes',[]) for c in lane};reserved|={physical(c) for c in cpus}
        reservations.append(dict(plan=str(path),cpus=sorted(cpus)))
    original=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
        if not line.strip().isdigit():continue
        pid=int(line.strip())
        try:cpus=sorted(os.sched_getaffinity(pid))
        except ProcessLookupError:continue
        original.append(dict(pid=pid,cpus=cpus))
        if len(cpus)<=16:reserved|={physical(c) for c in cpus}
    available={}
    for cpu in sorted(os.sched_getaffinity(0)):
        key=physical(cpu)
        if key not in reserved:available.setdefault(key,[]).append(cpu)
    lanes=list(available.values())[:10];assert len(lanes)==10
    free={int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
    gpus=sorted(free,key=lambda g:(-free[g],g))[:2];assert all(free[g]>8192 for g in gpus)
    def job(seed,index,smoke=False):
        ident=f'pick-and-place-ibolt-wide-H1.8-L2-vec256-utd0p125-s{seed}'
        return dict(id=ident,variant='wide',seed=seed,source=str(source),source_manifest=str(manifestpath),
            gpu=gpus[index],learner_cpus=lanes[index*5],worker_cpus=lanes[index*5+1:index*5+5],
            dacer_noise_scale=2.,dacer_target_entropy_per_dim=1.8,
            reference_learner_config=str(folder/'references/reference-learner.json'),
            reference_run_config=str(folder/'references/reference-run.json'),
            output=str(folder/('preflights' if smoke else 'runs')/ident),status='queued')
    plan=dict(campaign=a.campaign,host=4,source=str(source),python=str(root/'venv/bin/python'),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        native_files_verified=len(native),source_manifest=str(manifestpath),cpu_lanes=lanes,
        original_gpu_processes=original,cpu_lane_reservations=reservations,
        worker_module='rebuttal_tools.fetch_dacer_vector_worker',completion_marker='dacer-vector-verified-completion.json',
        extra_spec_fields=['dacer_noise_scale','dacer_target_entropy_per_dim','reference_learner_config','reference_run_config'],
        jobs=[job(0,0),job(1,1)],preflights=[job(100,0,True)],created_unix=time.time(),status='waiting_environment',
        config=dict(num_envs=256,batch_size=256,utd=.125,gradient_steps=32,warmup=12800,total_transitions=1000192,
            expected_updates=123424,temperature=.25,log_std_min=-5,log_std_max=1,initial_log_std=-1,
            target_entropy_per_dim=1.8,total_entropy_target=7.2,noise_scale=2.,alpha_lr=.03,interval_updates=10000),
        vector_control_reference=str(folder/'references/vector-control.json'),
        scope='Two-seed fast diagnostic. Compare to matched vector control; not a same-budget UTD1 SAC comparison.')
    path=folder/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    wrapper=folder/'wait-and-run.sh';wrapper.write_text(f'''#!/bin/bash
set -eu
while [ ! -f {root}/environment-ready ]; do sleep 5; done
exec {root}/venv/bin/python -u -m rebuttal_tools.fetch_sigma_vector_campaign --plan {path}
''')
    program=a.campaign+'-host4'
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
    for command in (['reread'],['add',program],['start',program]):
        subprocess.run(['sudo','-n','supervisorctl',*command],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(dict(program=program,plan=str(path),jobs=2,preflights=1,native_files_verified=len(native),source_commit=plan['source_commit'])),flush=True)


if __name__=='__main__':main()
