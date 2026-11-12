"""One process per free GPU; full campaign remains stopped until explicitly started."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .run import atomic, guard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    guard()
    plan = json.loads(a.plan.read_text())
    folder = a.plan.parent
    controller_lock = (folder/'controller.lock').open('a')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not plan['smoke']:
        verification = json.loads(Path(plan['verification']).read_text())
        assert verification['status'] == 'completed'
        assert json.loads(Path('/workspace/heechan-adroit/datasets/verified-demonstrations.json').read_text())['passed']
        assert len(verification['jobs']) == 8
        assert all(j['status'] == 'completed' for j in verification['jobs'])
        assert json.loads(Path('/workspace/heechan-adroit/environment-checks.json').read_text())['passed']
    children, locks = {}, {}
    try:
        while True:
            for pid, child in list(children.items()):
                if child.poll() is None:
                    continue
                job = next(j for j in plan['jobs'] if j.get('pid') == pid)
                marker = Path(job['output'])/'completed.json'
                if child.returncode == 0 and marker.exists():
                    result = json.loads(marker.read_text())
                    assert result['completed'] and result['smoke'] == plan['smoke']
                    assert (result['method'], result['task'], result['seed']) == (job['method'], job['task'], job['seed'])
                    assert result['steps'] == (13568 if plan['smoke'] else ((job['steps']+255)//256)*256)
                    job.update(status='completed', result=result)
                else:
                    failure = Path(job['output'])/'failure.json'
                    job.update(status='failed', error=failure.read_text() if failure.exists() else f'exit={child.returncode}')
                locks.pop(pid).close()
                child.wait()
                del children[pid]
            if all(j['status'] in ('completed', 'failed') for j in plan['jobs']):
                plan['status'] = 'completed_with_failures' if any(j['status'] == 'failed' for j in plan['jobs']) else 'completed'
                atomic(a.plan, plan)
                return
            # Query compute contexts, including processes managed by other campaigns.
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
            contexts = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                                '--format=csv,noheader,nounits'], text=True).splitlines()
            for line in raw.splitlines():
                gpu, uuid, memory, utilization = [v.strip() for v in line.split(',')]
                shared = plan.get('allow_shared_gpu', False)
                if (not shared and uuid in contexts) or int(memory) > (8192 if shared else 256) or int(utilization) > (60 if shared else 5):
                    continue
                if any(j.get('gpu') == int(gpu) and j['status'] == 'running' for j in plan['jobs']):
                    continue
                job = next((j for j in plan['jobs'] if j['status'] == 'queued'), None)
                if job is None:
                    break
                lockpath = Path('/workspace/.gmm40-gpu-queue-locks')/(uuid+'.lock')
                lockpath.parent.mkdir(exist_ok=True)
                lock = lockpath.open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                if Path(job['output']).exists():
                    lock.close()
                    raise RuntimeError('Refusing to overwrite existing run: '+job['output'])
                specpath = folder/'specs'/(job['id']+'.json')
                specpath.parent.mkdir(exist_ok=True)
                atomic(specpath, job)
                args = [plan['python'], '-u', '-m', 'adroit_rebuttal.run', '--spec', str(specpath)]
                if plan['smoke']:
                    args.append('--smoke')
                (folder/'logs').mkdir(exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                    XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl', WANDB_MODE='disabled',
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                with (folder/'logs'/(job['id']+'.log')).open('x') as stream:
                    child = subprocess.Popen(args, cwd=plan['source'], env=env,
                        stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                job.update(status='running', pid=child.pid, gpu=int(gpu), started_unix=time.time())
                children[child.pid], locks[child.pid] = child, lock
                print('Started '+job['id']+' on GPU'+gpu, flush=True)
            plan.update(status='running', counts=dict(Counter(j['status'] for j in plan['jobs'])),
                        observed_unix=time.time())
            atomic(a.plan, plan)
            time.sleep(5)
    except BaseException as error:
        plan.update(status='controller_error', error=repr(error))
        atomic(a.plan, plan)
        raise


if __name__ == '__main__':
    main()
