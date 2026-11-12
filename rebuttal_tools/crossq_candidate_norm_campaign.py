"""Finite paired EMA/current-BN campaign. Never resumes earlier experiments."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess as sp
import time


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['branch'] == 'heechan-rebuttal'
    assert plan['modes'] == ['ema', 'joint_bn']
    assert all(j['status'] == 'pending' for j in plan['jobs']), 'Refuse to restart a used plan'
    children = {}
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    plan['controller_pid'] = os.getpid()
    plan['status'] = 'running'
    save(args.plan, plan)
    while True:
        for j in plan['jobs']:
            if j['id'] not in children:
                continue
            child = children[j['id']]
            code = child.poll()
            if code is None:
                continue
            completions = list((Path(j['folder'])/'training').glob('*/completed.json'))
            valid = code == 0 and len(completions)==1 and json.loads(completions[0].read_text())['timesteps']==1_000_000
            j.update(status='completed' if valid else 'failed', exit_code=code, ended=time.time())
            children.pop(j['id'])
            print(json.dumps(dict(id=j['id'],status=j['status'],exit_code=code)),flush=True)
        if stopping:
            for j in plan['jobs']:
                if j['id'] in children:
                    os.killpg(children[j['id']].pid, signal.SIGTERM)
                    j['status'] = 'stop_requested'
                elif j['status']=='pending':
                    j['status']='cancelled'
            plan['status']='stopped';save(args.plan,plan)
            return
        if not any(j['status'] in ('pending','running') for j in plan['jobs']):
            plan['status']='completed' if all(j['status']=='completed' for j in plan['jobs']) else 'finished_with_failures'
            save(args.plan,plan)
            return
        gpu_rows = sp.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True)
        gpu = {int(row.split(',')[0]): tuple(float(x.strip()) for x in row.split(',')[1:])
               for row in gpu_rows.splitlines()}
        reservation = {}
        for j in plan['jobs']:
            if j['status']!='pending':
                continue
            free, util = gpu[j['gpu']]
            reserve = reservation.get(j['gpu'],0)
            if free-reserve < 6000:
                j['waiting_reason']='gpu_memory';continue
            if plan.get('wait_for_gpu_util_below') is not None and util >= plan['wait_for_gpu_util_below']:
                j['waiting_reason']='existing_gpu_work';continue
            if shutil.disk_usage(plan['root']).free < 3*2**30:
                j['waiting_reason']='disk_space';continue
            d=Path(j['folder']);d.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,PYTHONPATH=plan['source'],CUDA_VISIBLE_DEVICES=str(j['gpu']),
                CUDA_DEVICE_ORDER='PCI_BUS_ID',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='disabled',
                JAX_COMPILATION_CACHE_DIR=str(Path(plan['root'])/'jax-cache'))
            argv=['taskset','-c',','.join(map(str,j['cpus'])),plan['python'],'-u','-m',
                  'rebuttal_tools.crossq_candidate_norm','--mode',j['mode'],*j['overrides']]
            with (d/'stdout.log').open('w') as log:
                child=sp.Popen(argv,cwd=plan['source'],env=env,stdin=sp.DEVNULL,
                               stdout=log,stderr=sp.STDOUT,start_new_session=True)
            children[j['id']]=child
            j.update(status='running',pid=child.pid,started=time.time(),argv=argv,
                start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
            j.pop('waiting_reason',None)
            reservation[j['gpu']]=reserve+6000
            save(d/'launch.json',j)
            print(json.dumps(dict(id=j['id'],pid=child.pid,gpu=j['gpu'])),flush=True)
        plan['updated']=time.time()
        save(args.plan,plan)
        time.sleep(15)


if __name__=='__main__':
    main()
