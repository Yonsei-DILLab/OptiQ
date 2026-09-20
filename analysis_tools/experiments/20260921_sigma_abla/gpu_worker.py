"""One bounded 10k run per GPU at a time, managed by existing Supervisor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIGMAS = (.1, .2, .01, .05)
TASKS = {'180': ('humanoid','ant','halfcheetah'), '199': ('walker2d','hopper')}


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def main():
    host, gpu, root = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    root.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True).strip()
    sigma = SIGMAS[gpu]
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[gpu::4] or cpus)
    for task in TASKS[host]:
        name = f'{task}-trg-sigma{sigma:g}-10k-s0'
        path = root / (name+'.json')
        assert not path.exists(), 'Never silently repeat a previous attempt'
        args = [sys.executable, '-u', str(HERE/'train_sigma.py'), str(sigma),
                f'benchmark={task}', f'run_name={name}', f'output_root={root}/outputs']
        job = dict(task=task, sigma=sigma, seed=0, host=host, gpu=gpu,
                   commit=sha, command=args, status='running', started=time.time())
        save(path,job)
        with (root/(name+'.log')).open('x') as log:
            rc = subprocess.run(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        outputs = list((root/'outputs').glob(name+'_*/completed.json'))
        job.update(exit_code=rc, finished=time.time(), status='failed')
        if len(outputs)==1:
            result=json.loads(outputs[0].read_text())
            job.update(result, run_dir=str(outputs[0].parent))
            if result['timesteps']==10000 and rc==0:
                job['status']='completed'
        save(path,job)
        print(json.dumps(job), flush=True)


if __name__=='__main__':
    main()
