import hashlib,json,os,subprocess,sys,time
import importlib.metadata
import copy
from functools import lru_cache
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,optax
from flax.training.train_state import TrainState
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from optiq_dime.policy import ImplicitActor
from optiq_dime.algorithm import OptiQDIME
from common.type_aliases import RLTrainState

ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=32)
def config(seed=0):
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        cfg=compose(config_name='mujoco_setting',overrides=[f'seed={seed}'])
    return cfg

def begin(out,args):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    files=[p for folder in ('analysis_boltzmann','optiq_dime','common','diffusion','models','configs') for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.yaml')]
    versions={p:importlib.metadata.version(p) for p in ('jax','jaxlib','flax','optax','numpy','scipy','torch','matplotlib')}
    meta=dict(packages=versions,arguments=vars(args),time=time.time(),command=sys.argv,devices=[str(d) for d in jax.devices()],job=os.getenv('SLURM_JOB_ID'),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    try: meta['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError: meta['commit']='unknown'
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    cfg=copy.deepcopy(config(args.seed))
    cfg.env_name='FrozenQ' if hasattr(args,'case') else 'OptiQMoveCar-v0'
    if hasattr(args,'steps'): cfg.total_steps=args.steps
    if hasattr(args,'warmup'): cfg.alg.learning_starts=args.warmup; cfg.alg.actor.learning_starts=args.warmup
    cfg.wandb.activate=False
    (out/'resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    return out

def actor_state(dim,seed,obs_dim=1):
    cfg=config(seed); net=ImplicitActor(dim,tuple(cfg.alg.actor.hidden_dims))
    p=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,obs_dim)),jnp.zeros((1,dim)))['params']
    return TrainState.create(apply_fn=net.apply,params=p,tx=optax.adam(float(cfg.alg.optimizer.lr_actor),b1=.9,b2=.999))

def dummy_critic(fn):
    def apply(variables,obs,actions,**kwargs):
        v=fn(obs,actions)
        return jnp.stack((v,v),axis=0)[...,None]
    return RLTrainState.create(apply_fn=apply,params={},target_params={},batch_stats={},target_batch_stats={},tx=optax.adam(3e-4))

def ot_update(actor,critic,obs,key):
    # Exact production update, not a second implementation of OT.
    a=config().alg.actor
    return OptiQDIME.update_actor(actor,critic,obs,key,jnp.array([0.]),
        int(a.num_policy_samples),int(a.proposals_per_policy_sample),a.proposal_sampling_mode,
        float(a.proposal_std),float(a.proposal_clip),bool(a.include_anchor),bool(a.density_correction),
        float(a.density_beta),bool(a.adaptive_density_beta),float(a.minimum_source_ess),
        int(a.density_beta_grid_size),float(a.temperature),float(a.sinkhorn_epsilon),int(a.sinkhorn_iterations),a.source_q_eval,a.transport_target_mode)
