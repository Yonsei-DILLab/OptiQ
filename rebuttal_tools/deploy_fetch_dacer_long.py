"""Freeze a non-anonymous local partition of the fixed 10M DACER search."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.input.read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    source=root/('source-'+cfg['campaign']);source.mkdir(exist_ok=False)
    native=json.loads(Path(cfg['native_manifest']).read_text())
    with tarfile.open(cfg['native_archive']) as t:t.extractall(source,filter='data')
    with tarfile.open(cfg['code_archive']) as t:t.extractall(source,filter='data')
    if not (source/'.gitignore').exists():(source/'.gitignore').write_text('__pycache__/\n*.pyc\noutputs/\n*.log\n')
    for name,digest in native.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    for name,digest in json.loads(Path(cfg['collector_manifest']).read_text()).items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    subprocess.run([cfg['ibolt_python'],'-m','py_compile',str(source/'rebuttal_tools/fetch_dacer_long_worker.py'),
        str(source/'rebuttal_tools/fetch_dacer_long_campaign.py')],check=True)
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'],check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Freeze fixed 10M DACER search with paired SAC and milestone evaluations'],check=True,stdout=subprocess.DEVNULL)
    folder=root/'outputs'/cfg['campaign'];folder.mkdir(parents=True,exist_ok=False)
    for name in ('specs','logs','runs','preflights','references','mpl-cache'):(folder/name).mkdir()
    cache=Path(cfg.get('jax_cache_reference','/nonexistent'))
    if cache.is_dir():(folder/'jax-cache').symlink_to(cache,target_is_directory=True)
    else:(folder/'jax-cache').mkdir()
    ref=folder/'references'/'ibolt-winning-config.json';shutil.copyfile(cfg['ibolt_reference'],ref)
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    manifestpath=folder/'source-sha256.json';manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native,indent=2)+'\n')
    def job(method,profile,target,scale,seed,phase):
        ident=profile+(f'-preflight' if phase=='preflight' else f'-s{seed}')
        return dict(id=ident,profile=profile,method=method,seed=seed,phase=phase,
            target_entropy_per_dim=target,noise_scale=scale,source=str(source),source_manifest=str(manifestpath),
            python=cfg['sac_python'] if method=='sac' else cfg['ibolt_python'],ibolt_reference_config=str(ref),
            output=str(folder/('preflights' if phase=='preflight' else 'runs')/ident),status='queued')
    search=[];preflights=[]
    for profile in cfg['profiles']:
        key,h,l=profile['id'],profile['target'],profile['scale']
        preflights.append(job('ibolt',key,h,l,100,'preflight'))
        search.extend(job('ibolt',key,h,l,s,'search') for s in (0,1))
    baselines=[job('sac','SAC',-.9,2.,s,'baseline') for s in cfg.get('sac_seeds',[])]
    if baselines:preflights.append(job('sac','SAC',-.9,2.,100,'preflight'))
    # Start the paired SAC pilot and first iBOLT profile together when capacity is tight.
    jobs=baselines[:2]+search+baselines[2:]
    plan=dict(cfg,campaign=cfg['campaign'],source=str(source),source_manifest=str(manifestpath),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        native_files_verified=len(native),jobs=jobs,preflights=preflights,selection=None,
        created_unix=time.time(),status='prepared',
        protocol=dict(num_envs=256,utd=.125,batch_size=256,warmup=12800,steps=10000128,target_steps=10000000,
            expected_updates=1248416,HER=True,normalization=True,temperature=.25,logstd=[-5,1],initial_logstd=-1,
            milestones=[5000192,10000128],milestone_episodes=500,final_episodes=500,
            search_seeds=[0,1],confirmation_seeds=[2,3,4],no_early_stopping=True,
            comparison='Same transitions, optimizer updates, architecture, learning rates, HER and normalization; tuned iBOLT vs unchanged off-the-shelf SAC. All search results retained.'))
    if cfg.get('reuse_plan'):
        previous=json.loads(Path(cfg['reuse_plan']).read_text());oldsource=Path(previous['source'])
        assert previous['status'] in ('stopped','controller_error','completed','completed_with_failures')
        assert (oldsource/'rebuttal_tools/fetch_dacer_long_worker.py').read_bytes()==(source/'rebuttal_tools/fetch_dacer_long_worker.py').read_bytes()
        plan['cpu_checks_passed']=bool(previous.get('cpu_checks_passed'))
        if plan['cpu_checks_passed']:
            shutil.copyfile(Path(cfg['reuse_plan']).parent/'logs/cpu-checks.log',folder/'logs/cpu-checks.log')
        old={j['profile']:j for j in previous['preflights']}
        for j in preflights:
            candidate=old.get(j['profile'])
            if not candidate:continue
            marker=Path(candidate['output'])/'verified-completion.json'
            if not marker.exists():continue
            result=json.loads(marker.read_text())
            assert result['completed'] and result['smoke'] and result['updates']==96 and result['observed_utd']==.125
            assert result['target_entropy_per_dim']==(j['target_entropy_per_dim'] if j['method']=='ibolt' else None)
            assert result['noise_scale']==(j['noise_scale'] if j['method']=='ibolt' else None)
            j.update(status='completed',output=candidate['output'],result=result,verification_reused_from=cfg['reuse_plan'])
    path=folder/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    program=cfg['campaign']+f'-host{cfg["host"]}'
    conf=f'''[program:{program}]
command={cfg['ibolt_python']} -u -m rebuttal_tools.fetch_dacer_long_campaign --plan {path}
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
    for command in (['reread'],['add',program],['start',program]):
        subprocess.run(['sudo','-n','supervisorctl',*command],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(dict(host=cfg['host'],program=program,plan=str(path),source=str(source),
        source_commit=plan['source_commit'],native_files_verified=len(native),search_runs=len(search),
        baseline_runs=len(baselines),additional_confirmation_runs=3,preflights=len(preflights))),flush=True)


if __name__=='__main__':main()
