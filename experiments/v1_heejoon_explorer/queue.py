"""One GPU per environment, four sequential seeds; GPU 3 is explicitly unused."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAP={'ant':0,'humanoid':1,'halfcheetah':2}


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=MAP,required=True)
    p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();repo=Path(__file__).resolve().parents[2]
    runtime=root/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    validation=json.loads((root/'validation/VALIDATION_PASSED.json').read_text())
    manifest=json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    assert validation['passed'] and validation['commit']==manifest['commit']
    lock=(runtime/f'{args.env}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(MAP[args.env])
    env['WANDB_API_KEY']=(Path.home()/'.config/optiq-secrets/wandb_api_key').read_text().strip()
    env.update(WANDB_MODE='online',WANDB_ENTITY='OptiQ',WANDB_PROJECT='legacy_explorer',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',
               JAX_COMPILATION_CACHE_DIR=str(root/'jax_cache'),PYTHONPATH=str(repo))
    env.pop('LD_LIBRARY_PATH',None)
    # Give each worker two CPU cores, disjoint where possible; no CPU oversubscription.
    available=sorted(os.sched_getaffinity(0));offset=2*MAP[args.env]
    cpus=available[offset:offset+2] or available[:2]
    for seed in range(4):
        out=root/'runs'/f'{args.env}_s{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'COMPLETE.json').exists():continue
        if (runtime/'STOP_NEW_RUNS').exists():break
        command=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-m',
                 'experiments.v1_heejoon_explorer.run','--env',args.env,'--seed',str(seed),'--out',str(out)]
        with (out/'console.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            record=dict(env=args.env,seed=seed,pid=proc.pid,gpu=MAP[args.env],cpus=cpus,
                        commit=manifest['commit'],started=time.time(),state='running')
            (runtime/f'{args.env}.json').write_text(json.dumps(record,indent=2)+'\n')
            code=proc.wait()
        record.update(exit_code=code,finished=time.time(),state='complete' if (out/'COMPLETE.json').exists() else 'paused' if code==0 else 'failed')
        (runtime/f'{args.env}.json').write_text(json.dumps(record,indent=2)+'\n')
        # A failure is visible and does not silently consume all remaining seeds.
        if record['state']!='complete':return
    (runtime/f'{args.env}_QUEUE_DONE').write_text('All eligible seeds processed.\n')


if __name__=='__main__':main()
