"""Freeze and supervise a five-seed suite partition on servers 2, 3 or 4."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.input.read_text())
    assert cfg['host']==4, 'This alpha sweep runs only on server4; server1 stays reserved'
    root = Path(cfg['root'])
    root.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(root).free > 50*1024**3, 'Insufficient disk headroom'
    source = root/('source-'+cfg['campaign'])
    source.mkdir(exist_ok=False)
    native = json.loads(Path(cfg['native_manifest']).read_text())
    for archive in ('native_archive','code_archive'):
        with tarfile.open(cfg[archive]) as stream:
            stream.extractall(source, filter='data')
    for name, digest in native.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    (source/'.gitignore').write_text('__pycache__/\n*.pyc\noutputs/\n*.log\n')
    subprocess.run(['git','init','-b','heechan-rebuttal',str(source)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(source),'add','.'], check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Heechan','-c','user.email=jason02951234@gmail.com',
        'commit','-m','Freeze fixed-alpha SAC Push sweep with two-seed selection and confirmation'], check=True, stdout=subprocess.DEVNULL)
    folder = root/'outputs'/cfg['campaign']
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('specs','logs','runs','preflights','references','mpl-cache','jax-cache'):
        (folder/name).mkdir()
    manifest = {str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts}
    manifest_path = folder/'source-sha256.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    (folder/'native-source-sha256.json').write_text(json.dumps(native, indent=2)+'\n')
    def job(alpha,seed,phase):
        ident=f'push-sac-alpha{alpha:g}-s{seed}'
        return dict(id=ident,method='sac',task='push',alpha=alpha,seed=seed,phase=phase,
            source=str(source),source_manifest=str(manifest_path),python=cfg['torch_python'],
            output=str(folder/('preflights' if phase=='preflight' else 'runs')/ident),status='queued')
    preflights=[job(alpha,100,'preflight') for alpha in cfg['alphas']]
    jobs=[job(alpha,seed,'search') for seed in (0,1) for alpha in cfg['alphas']]
    plan = dict(cfg, source=str(source), source_manifest=str(manifest_path),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip(),
        native_files_verified=len(native), jobs=jobs, preflights=preflights,
        created_unix=time.time(), status='prepared',
        protocol=dict(num_envs=256,utd=.125,batch_size=256,warmup=12800,
            actual_steps=5000192,target_steps=5000000,expected_updates=623424,
            gradient_steps=32,HER='future 204/256 in each minibatch',
            normalization='online real observations only, clip5, no reward normalization',
            eval_every=5000,eval_episodes=10,final_episodes=100,
            checkpoint_every=100000,replay_capacity=1000000,seeds=[0,1,2,3,4],
            entropy_mode='fixed',alphas=cfg['alphas'],pilot_seeds=[0,1],confirmation_seeds=[2,3,4],
            selection='Pilot last5 at 5M, then AUC; final100 excluded from selection',
            reporting='5-seed last5 and final100 success means, seed SD, full curves; no seed selection'))
    if cfg.get('reuse_cpu_plan'):
        previous_path = Path(cfg['reuse_cpu_plan'])
        previous = json.loads(previous_path.read_text())
        assert previous['cpu_checks_passed']
        for name in ('check_fetch_suite_vector.py','fetch_suite_vector_env.py','fetch_suite_vector_replay.py'):
            relative = Path('rebuttal_tools')/name
            assert (Path(previous['source'])/relative).read_bytes() == (source/relative).read_bytes()
        plan['cpu_checks_passed'] = True
        plan['cpu_verification_reused_from'] = str(previous_path)
        shutil.copyfile(previous_path.parent/'logs/cpu-checks.log', folder/'logs/vector-her-previous-checks.log')
    with (folder/'logs/cpu-checks.log').open('w') as stream:
        subprocess.run([cfg['torch_python'],'-u','-m','rebuttal_tools.check_fetch_sac_alpha'],
            cwd=source,env=dict(__import__('os').environ,PYTHONPATH=str(source)),stdout=stream,stderr=subprocess.STDOUT,check=True)
    plan['cpu_checks_passed']=True
    path = folder/'plan.json'
    path.write_text(json.dumps(plan, indent=2)+'\n')
    program = cfg['campaign']+f'-host{cfg["host"]}'
    conf = f'''[program:{program}]
command={cfg['ibolt_python']} -u -m rebuttal_tools.fetch_sac_alpha_campaign --plan {path}
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
    subprocess.run(['sudo','-n','tee',f'/etc/supervisor/conf.d/{program}.conf'], input=conf,
        text=True, check=True, stdout=subprocess.DEVNULL)
    for command in (['reread'],['add',program],['start',program]):
        subprocess.run(['sudo','-n','supervisorctl',*command], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps(dict(host=cfg['host'], program=program, plan=str(path), source=str(source),
        source_commit=plan['source_commit'], native_files_verified=len(native),
        main_jobs=len(jobs), preflights=len(preflights))), flush=True)


if __name__ == '__main__':
    main()
