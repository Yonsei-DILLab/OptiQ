"""Freeze equal-budget entropy trials without touching existing learners."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.input.read_text());assert cfg['host'] in (2,3,4)
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(root).free>50*1024**3
    source=root/('source-'+cfg['campaign']);source.mkdir(exist_ok=False)
    native=json.loads(Path(cfg['native_manifest']).read_text())
    for name in ('native_archive','code_archive'):
        with tarfile.open(cfg[name]) as t:t.extractall(source,filter='data')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\noutputs/\n*.log\n')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Freeze fair Fetch entropy grids with disjoint pilot and final seeds'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/cfg['campaign'];folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','runs','logs','preflights','references','jax-cache','mpl-cache'):(folder/name).mkdir()
    reference=folder/'references/ibolt-winning-config.json';shutil.copyfile(cfg['ibolt_reference'],reference)
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    manifest_path=folder/'source-sha256.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    sys.path.insert(0,str(source))
    from rebuttal_tools.fetch_entropy_campaign import prepare_jobs,GRIDS
    plan=dict(cfg,folder=str(folder),source=str(source),source_manifest=str(manifest_path),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        native_files_verified=len(native),ibolt_reference_config=str(reference),selections={},created_unix=time.time(),status='prepared',
        grids=GRIDS,protocol=dict(tasks=['slide','reach'],pilot_seeds=[5,6],fresh_final_seeds=[0,1,2,3,4],
            configurations_per_method_per_task=3,pilot_trials_global=48,final_trials_global=40,
            target_steps=5000000,actual_steps=5000192,expected_updates=623424,
            num_envs=256,utd=.125,batch_size=256,gradient_steps=32,warmup=12800,
            HER='future 204/256 in every optimizer minibatch',
            normalization='Original real-observation RMS only, clip5; no reward normalization',
            eval_every=5000,eval_episodes=10,final_episodes=100,checkpoint_every=100000,
            selection='Per environment/method: mean last5 regular evaluations over seeds5/6; ties final100, full-curve mean, worst-seed last5, existing default proximity, fixed grid order.',
            failed_trials='Exclude incomplete parameter pairs; no replacement seeds or extra configurations. All-zero valid searches promote the preset tie-break winner, labeled unvalidated.',
            final_runs='Five completely new initializations, seeds0..4; no pilot checkpoint or earlier-run reuse',
            final_reporting='Seed mean/sample SD, fixed-budget last5 and independent final100 kept separate',
            fairness='Equal trial counts, transition/update/evaluation budgets. Shared-GPU wall time is not a matched compute-time measurement. Architecture/LR/other method settings retained from existing Fetch5M snapshots.'))
    prepare_jobs(plan);path=folder/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH=str(source)+':'+cfg.get('extra_pythonpath',''),CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    check=subprocess.run([cfg['controller_python'],'-u','-m','rebuttal_tools.check_fetch_entropy','--methods',*cfg['methods']],
        cwd=source,env=env,capture_output=True,text=True)
    (folder/'logs/cpu-checks.log').write_text(check.stdout+'\n'+check.stderr)
    assert check.returncode==0,check.stdout[-2000:]+check.stderr[-4000:]
    audit=json.loads(check.stdout.strip().splitlines()[-1]);assert audit['passed'] and all(r['learner_updates']==0 for r in audit['cases'])
    (folder/'cpu-checks.json').write_text(json.dumps(audit,indent=2)+'\n')
    plan['cpu_checks_passed']=True;path.write_text(json.dumps(plan,indent=2)+'\n')
    # Sidecars reload policies each admission cycle; include initializing workers.
    for filename in cfg.get('capacity_policies',[]):
        policy_path=Path(filename);policy=json.loads(policy_path.read_text())
        (folder/('before-policy-'+policy_path.parent.name+'.json')).write_text(json.dumps(policy,indent=2)+'\n')
        if str(path) not in policy['peer_plans']:policy['peer_plans'].append(str(path))
        temp=policy_path.with_suffix('.entropy.tmp');temp.write_text(json.dumps(policy,indent=2)+'\n');temp.replace(policy_path)
    program=cfg['campaign']+f'-host{cfg["host"]}'
    conf=f'''[program:{program}]
command={cfg['controller_python']} -u -m rebuttal_tools.fetch_entropy_campaign --plan {path}
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
    subprocess.run(['sudo','-n','tee',f'/etc/supervisor/conf.d/{program}.conf'],input=conf,text=True,check=True,stdout=subprocess.DEVNULL)
    for command in (['reread'],['add',program],['start',program]):subprocess.run(['sudo','-n','supervisorctl',*command],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(dict(host=cfg['host'],program=program,plan=str(path),source=str(source),source_commit=plan['source_commit'],
        native_files_verified=len(native),pilot_runs=len(plan['jobs']),fresh_final_runs_after_selection=len(cfg['methods'])*10,
        gpu_preflights=len(plan['preflights']),cpu_cases=len(audit['cases']),methods=cfg['methods'])))


if __name__=='__main__':main()
