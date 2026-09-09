#!/usr/bin/env python3
"""Execute a fixed hyperparameter manifest on one GPU via supervisor."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('manifest', type=Path)
p.add_argument('worker', type=int, choices=range(4))
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.worker))
child = None
stop_requested = False
def stop(signum, frame):
    global stop_requested
    stop_requested = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
for dependency in manifest.get('launch_dependencies', {}).get(str(a.worker), []):
    last_status = None
    while not stop_requested:
        status = subprocess.run(['supervisorctl', 'status', dependency['program']],
                                capture_output=True, text=True).stdout.strip()
        if status != last_status:
            print(json.dumps({'waiting_for': dependency['program'], 'status': status}), flush=True)
            last_status = status
        fields = status.split()
        if len(fields) >= 2 and fields[1] == 'EXITED':
            completion = Path(dependency['completion'])
            if not completion.exists() or not json.loads(completion.read_text()).get('finished'):
                raise RuntimeError('Dependency exited without a completed training run: '+dependency['program'])
            break
        time.sleep(30)
    if stop_requested:
        sys.exit(0)
for index, job in enumerate(manifest['jobs']):
    if stop_requested:
        break
    if index % 4 != a.worker:
        continue
    if not job.get('enabled', True):
        continue
    output = ROOT/'outputs/gmm40_tuning'/manifest['phase']/f"{index:03d}-{job['label']}"
    if list(output.glob('*/completed.json')):
        print(f'Skipping completed job {index}: {output}', flush=True)
        continue
    command = [str(ROOT/'scripts/run_gmm40.sh'), '--output-root', str(output),
               '--group', manifest.get('group', 'gmm40-hyperparameters-'+manifest['phase']),
               '--job-type', manifest.get('job_type', 'hyperparameter-search')]
    for key, value in job.items():
        if key in ('label', 'enabled'):
            continue
        flag = '--'+key.replace('_','-')
        if isinstance(value, bool):
            command.append(flag if value else '--no-'+key.replace('_','-'))
        elif isinstance(value, list):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    print(json.dumps({'job_index':index, 'job':job, 'command':command}), flush=True)
    child = subprocess.Popen(command, cwd=ROOT, env=env)
    returncode = child.wait()
    if stop_requested:
        break
    if returncode:
        print(f'Job {index} failed with exit {returncode}; retained logs', flush=True)
        sys.exit(returncode)
