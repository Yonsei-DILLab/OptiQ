"""Operational W&B startup repair; execute an unchanged numerical snapshot.

Only replace its verify_online function. Source checks, algorithms, checkpoints,
seed, output directory and W&B ID remain owned by the original runner.
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
import uuid
from pathlib import Path


def verify_upload(run, step, api_factory, timeout=300, clock=time.monotonic,
                  sleep=time.sleep, token=None):
    token = token or uuid.uuid4().hex
    start = clock()
    deadline = start + timeout
    polls = 0
    last_error = None
    last_send = float('-inf')
    api = api_factory()
    while clock() < deadline:
        # Summary assignment alone can remain buffered until training or finish.
        # Commit a real history row, without advancing the environment-step axis.
        if clock() - last_send >= 30:
            run.log({'ops/upload_probe': token,
                     'ops/upload_probe_env_step': int(step)}, commit=True)
            last_send = clock()
        try:
            api.flush()  # Public API cache invalidation, NOT an SDK upload flush.
            remote = api.run(f'{run.entity}/{run.project}/{run.id}')
            polls += 1
            if remote.summary.get('ops/upload_probe') == token:
                return dict(verified=True, polls=polls,
                            elapsed_seconds=clock()-start, env_step=int(step))
        except Exception as exc:
            last_error = type(exc).__name__  # Never record auth/request payloads.
        sleep(min(5, max(0, deadline-clock())))
    raise RuntimeError(f'W&B committed-history upload unconfirmed at env step {step}; '
                       f'timeout={timeout}s, polls={polls}, last_error={last_error}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--module',required=True,choices=['experiments.v3_heejoon_explorer.run','experiments.v4_heejoon_explorer.run'])
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--env',required=True,choices=['ant','humanoid','halfcheetah'])
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    package=Path(__file__).parent
    manifest=json.loads((package/'SCHEDULER_MANIFEST.json').read_text())
    assert all(hashlib.sha256((package/f).read_bytes()).hexdigest()==h
               for f,h in manifest['files'].items())
    # The operational package also contains queue.py; do not shadow Python's
    # standard-library queue when JAX/Torch/W&B import it below.
    sys.path[:]=[x for x in sys.path if Path(x).resolve()!=package.resolve()]
    sys.path.insert(0,str(a.source_root.resolve()))
    runner=importlib.import_module(a.module)
    original=runner.verify_source()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    runner.write(out/'LAUNCHER.json',dict(numerical_commit=original['commit'],
                  launcher_commit=manifest['commit'],source_root=str(a.source_root),
                  change='verify_online only: committed history, 300s bounded polling',
                  time=time.time()))
    def verified(run,step):
        import wandb
        receipt=verify_upload(run,step,lambda:wandb.Api(timeout=15))
        receipt.update(time=time.time(),run_id=run.id,launcher_commit=manifest['commit'])
        with (out/'ONLINE_HEALTH.jsonl').open('a') as f:
            f.write(json.dumps(receipt)+'\n')
        print('ONLINE_HEALTH',json.dumps(receipt),flush=True)
    runner.verify_online=verified
    runner.run(a)


if __name__=='__main__':main()
