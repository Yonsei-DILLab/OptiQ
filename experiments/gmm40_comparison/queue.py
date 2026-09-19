"""One supervised GPU-3 worker; round robin 15K checkpoints, never touch GPU0–2."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from .run import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();repo=Path(__file__).resolve().parents[2];runtime=root/'runtime';runtime.mkdir(exist_ok=True,parents=True)
    lock=(runtime/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((repo/'experiments/gmm40_comparison/plan.json').read_text());manifest=json.loads((repo/'SOURCE_MANIFEST.json').read_text())
    val=json.loads((root/'validation/VALIDATION_PASSED.json').read_text());assert val['passed'] and val['commit']==manifest['commit']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        JAX_DEFAULT_MATMUL_PRECISION='highest',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda',PYTHONUNBUFFERED='1',PYTHONPATH=str(repo),WANDB_MODE='online',
        JAX_COMPILATION_CACHE_DIR=str(root/'jax_cache'))
    env['WANDB_API_KEY']=(Path.home()/'.config/optiq-secrets/wandb_api_key').read_text().strip();env.pop('LD_LIBRARY_PATH',None)
    order=['legacy_matched','v5_ot_matched','gmm_matched','monge_matched','gmm_small','gmm_large','v5_ot_large','legacy_reference']
    for end in range(plan['chunk'],plan['updates']+1,plan['chunk']):
        for seed in plan['seeds']:
            for name in order:
                if (runtime/'STOP').exists():return
                out=root/'runs'/f'{name}_s{seed}';out.mkdir(parents=True,exist_ok=True)
                if (out/'FAILED.json').exists():continue
                if (out/'state.json').exists() and json.loads((out/'state.json').read_text())['step']>=end:continue
                cmd=['taskset','-c','6-9',sys.executable,'-m','experiments.gmm40_comparison.run','--condition',name,'--seed',str(seed),'--until',str(end),'--root',str(root)]
                with (out/'console.log').open('a') as log:
                    proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                    state=dict(condition=name,seed=seed,until=end,pid=proc.pid,gpu=3,commit=manifest['commit'],started=time.time(),state='running')
                    write_json(runtime/'queue.json',state);rc=proc.wait()
                state.update(exit_code=rc,finished=time.time(),state='segment_complete' if rc==0 else 'failed')
                write_json(runtime/'queue.json',state)
                with (runtime/'segments.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
                if rc!=0:
                    write_json(out/'FAILED.json',dict(exit_code=rc,time=time.time()))
                    # Failures are retained, other independent conditions remain runnable.
    write_json(runtime/'QUEUE_FINISHED.json',dict(time=time.time(),commit=manifest['commit']))


if __name__=='__main__':main()
