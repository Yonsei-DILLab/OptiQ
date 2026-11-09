"""Deploy the revised two-seed search without changing live learner sources."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import time

TASKS = {1: ['walker2d', 'halfcheetah'], 2: ['ant'], 3: ['humanoid', 'hopper']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--old-folder', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=source, text=True).strip()
    assert revision == '262e873b0d17252aebea3f6303120fc309bac0c6'
    assert branch and 'anonymous' not in branch.lower()
    sys.path.insert(0, str(source))
    from rebuttal_tools.mujoco_logstd_campaign import make_plan
    from rebuttal_tools.mujoco_logstd_worker import atomic_json
    from importlib.util import spec_from_file_location, module_from_spec
    module_spec = spec_from_file_location('staged_controller', args.controller)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'runs', 'preflights', 'specs', 'selections'):
        (folder/name).mkdir()
    used = Counter()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if 'python' in (proc/'comm').read_text():
                cpus = os.sched_getaffinity(int(proc.name))
                if len(cpus) <= 8:
                    used.update(cpus)
        except OSError:
            pass
    allowed = os.sched_getaffinity(0)
    lanes = set()
    for cpu in allowed:
        siblings = set()
        for part in Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
            if '-' in part:
                lo, hi = map(int, part.split('-'))
                siblings.update(range(lo, hi+1))
            else:
                siblings.add(int(part))
        lanes.add(tuple(sorted(siblings & allowed)))
    lanes = sorted(lanes, key=lambda c: (sum(used[x] for x in c), c))[:{1:32, 2:48, 3:80}[args.host]]
    plan = make_plan(folder, source, args.python, 1, revision, [list(c) for c in lanes])
    plan.update(host=args.host, folder=str(folder), campaign=folder.parent.name,
        tasks=TASKS[args.host], temperatures=[.1, .25, .5], search_seeds=[0, 1],
        confirmation_seeds=[2, 3, 4], selections={}, jobs=[],
        gpu_reserve_mib=2560, created_unix=time.time(),
        selection_rule='Mean over seeds 0,1 of within-seed last 5 native stochastic_z evaluations at 980k..1M; ties choose lower temperature.')
    for task in plan['tasks']:
        for utd in (1, 2):
            for temperature in module.GRID:
                for seed in (0, 1):
                    plan['jobs'].append(module.make_job(plan, task, utd, temperature, seed))
    adopted = []
    if args.old_folder:
        old = json.loads((args.old_folder/'plan.json').read_text())
        handoff = json.loads((args.old_folder/'handoff-protocol-revision.json').read_text())
        assert old['status'] == 'superseded_by_two_seed_temp_grid'
        plan['preflight'] = old['preflight']
        plan['prior_campaign'] = str(args.old_folder)
        by_id = {j['id']: j for j in handoff['kept']}
        for j in plan['jobs']:
            if j['id'] in by_id:
                prior = by_id[j['id']]
                assert (prior['task'], prior['seed'], prior['utd'], prior['temperature']) == (j['task'], j['seed'], j['utd'], j['temperature'])
                j.update(prior, phase='search', adopted_from=str(args.old_folder))
                adopted.append(j['id'])
        assert len(adopted) == 4
    controller = folder/'controller.py'
    shutil.copyfile(args.controller, controller)
    plan['controller_sha256'] = hashlib.sha256(controller.read_bytes()).hexdigest()
    atomic_json(folder/'plan.json', plan)
    program = plan['campaign']+f'-host{args.host}'
    config = f'''[program:{program}]
command={args.python} -u {controller} --plan {folder}/plan.json
directory={source}
user={pwd.getpwuid(os.getuid()).pw_name}
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
    subprocess.run(['sudo', '-n', 'tee', f'/etc/supervisor/conf.d/{program}.conf'], input=config,
        text=True, stdout=subprocess.DEVNULL, check=True)
    subprocess.run(['sudo', '-n', 'supervisorctl', 'reread'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['sudo', '-n', 'supervisorctl', 'add', program], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['sudo', '-n', 'supervisorctl', 'start', program], check=True)
    print(json.dumps(dict(host=args.host, program=program, initial_jobs=len(plan['jobs']),
        adopted_jobs=adopted, confirmation_jobs_after_selection=len(plan['tasks'])*6,
        plan=str(folder/'plan.json'), source_commit=revision)), flush=True)


if __name__ == '__main__':
    main()
