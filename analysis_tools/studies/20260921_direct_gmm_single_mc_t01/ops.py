"""Source checks and commands; no credentials in manifests or command lines."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
ORDER = ['humanoid','halfcheetah']
GROUP = '20260921_DirectGMM_SingleQ_MC64_N64_M64_T01'


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def verify():
    launch=read(ROOT/'DEPLOYMENT.json')
    for rel,h in launch['files'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    return launch


def environment(gpu):
    e=os.environ.copy()
    for k in ['WANDB_RUN_ID','WANDB_RESUME','WANDB_NAME','PYTHONPATH','JAX_DEFAULT_MATMUL_PRECISION','LD_LIBRARY_PATH']:
        e.pop(k,None)
    e.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',MUJOCO_GL='egl',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        WANDB_MODE='online',WANDB_ENTITY='OptiQ',WANDB_PROJECT='DirectGMM_heejoon')
    e['WANDB_API_KEY']=(Path.home()/'.config/optiq-secrets/wandb_api_key').read_text().strip()
    return e


def command(env,seed,commit,smoke=False):
    assert env in ORDER and seed in range(4)
    out=ROOT/('validation' if smoke else 'runs')/f'{env}_s{seed}'
    args=[sys.executable,str(ROOT/'repo/run_optiq_dime.py'),
        '--config-name=mujoco_direct_gmm_single_mc',f'benchmark={env}',f'seed={seed}',
        f'output_root={out}',f'+experiment_commit={commit}',
        f'wandb.group={GROUP}'+('_validation' if smoke else ''),
        'run_name='+('validation-' if smoke else '')+f'{env}-DirectGMM-SingleQ-MC64-N64-M64-T01-s{seed}']
    if smoke:
        args+=['total_steps=128','alg.learning_starts=32','alg.actor.learning_starts=32',
               'alg.buffer_size=256','eval_interval=128','num_eval_episodes=1','checkpoint_interval=64',
               'diagnostic_interval=64','wandb.job_type=configuration-validation']
    return args
