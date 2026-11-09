"""Create an isolated copy of the existing Fetch source; prepare SAC batch ablation."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

folder = Path(__file__).resolve().parent
root = Path('/workspace/iBOLT')
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip() == 'heechan-rebuttal'
old = root/'outputs/fetch-rebuttal/pick-and-place-20261108T140900Z/host1/source'
source = folder/'source'
assert not source.exists(), 'Never overwrite a campaign source'
shutil.copytree(old, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(folder/'run.py', source/'fetch_rebuttal/run.py')
manifest = source/'fetch_rebuttal/SOURCE.json'
data = json.loads(manifest.read_text())
data['files']['run.py'] = hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
manifest.write_text(json.dumps(data, indent=2)+'\n')
changed = []
for p in old.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
        relative = p.relative_to(old)
        if p.read_bytes() != (source/relative).read_bytes():
            changed.append(str(relative))
assert sorted(changed) == ['fetch_rebuttal/SOURCE.json', 'fetch_rebuttal/run.py'], changed
usage = Counter()
for p in Path('/proc').glob('[0-9]*'):
    try:
        if 'python' in (p/'comm').read_text():
            affinity = os.sched_getaffinity(int(p.name))
            if len(affinity) <= 8:
                usage.update(affinity)
    except (OSError, ProcessLookupError):
        pass
allowed = os.sched_getaffinity(0)
lanes = set()
for cpu in allowed:
    siblings = set()
    for part in Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-')); siblings.update(range(lo, hi+1))
        else:
            siblings.add(int(part))
    lanes.add(tuple(sorted(siblings & allowed)))
lanes = sorted(lanes, key=lambda c: (sum(usage[x] for x in c), c))[:1]
assert len(lanes) == 1
for name in ('logs', 'runs', 'preflights'):
    (folder/name).mkdir(exist_ok=True)
python = '/workspace/iBOLT-runs/fetch-20261108-venv/bin/python'
jobs = [dict(id=f'pick-and-place-sac-b2048-s{s}', seed=s, status='queued',
             output=str(folder/'runs'/f'pick-and-place-sac-b2048-s{s}')) for s in (0,)]
plan = dict(source=str(source), python=python, host=1, jobs=jobs, cpu_lanes=lanes,
    allowed_gpus=[0, 1, 2, 3], max_admission_util=88, status='prepared',
    preflight=dict(id='sac-b2048-smoke', seed=100, smoke=True, status='queued',
                   output=str(folder/'preflights/sac-b2048')),
    protocol=dict(task='FetchPickAndPlace-v4', method='sac', seeds=[0],
        batch_size=2048, utd=1, steps=1000000, warmup=5000, her=False,
        observation_normalization=False, start='fresh, no checkpoint warm start',
        final_eval_episodes=100, checkpoint_every=100000),
    reference_source=str(old), changes_from_reference=changed,
    evaluation='same deterministic SAC evaluation as batch256',
    created_unix=time.time(), source_hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*') if p.is_file()})
(folder/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
program = 'fetch-sac-b2048-utd1-20261109T044224Z'
conf = f'''[program:{program}]
command={python} -u {folder}/controller.py
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
subprocess.run(['sudo','tee',f'/etc/supervisor/conf.d/{program}.conf'], input=conf, text=True,
               stdout=subprocess.DEVNULL, check=True)
subprocess.run(['sudo','supervisorctl','reread'], check=True, stdout=subprocess.DEVNULL)
subprocess.run(['sudo','supervisorctl','add',program], check=True, stdout=subprocess.DEVNULL)
print(json.dumps(dict(prepared=True, program=program, folder=str(folder), jobs=len(jobs),
                      cpu_lanes=lanes, changes=changed)), flush=True)
