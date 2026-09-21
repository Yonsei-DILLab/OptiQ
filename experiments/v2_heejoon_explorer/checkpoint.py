"""Atomic latest checkpoint including evaluator, replay, RNG and simulator."""
import io
import os
import pickle
import random
import zipfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import torch
import mujoco

MODEL_FIELDS = ['num_timesteps','_n_updates','_episode_num','_last_obs','_last_original_obs',
                '_last_episode_starts','ep_info_buffer','ep_success_buffer','_current_progress_remaining']
CALLBACK_FIELDS = ['n_calls','num_timesteps','best_mean_reward','last_mean_reward',
                   'evaluations_results','evaluations_timesteps','evaluations_length',
                   'evaluations_successes','evaluations_solved_steps']
WRAPPER_FIELDS = ['_elapsed_steps','rewards','needs_reset','total_steps','episode_returns',
                  'episode_lengths','episode_times','current_reset_info','return_queue',
                  'length_queue','episode_count','episode_start_time']
REPLAY_FIELDS = ['observations','next_observations','actions','rewards','dones','timeouts']


def states(model):
    return dict(explorer=model.policy.actor_state, evaluator=model.policy.target_actor_state,
                critic=model.policy.qf_state, entropy=model.ent_coef_state)


def simulator(vec):
    base = vec.envs[0].unwrapped
    wrappers = []; env = vec.envs[0]
    while True:
        wrappers.append({k:v for k,v in env.__dict__.items() if k in WRAPPER_FIELDS})
        if not hasattr(env, 'env'): break
        env = env.env
    return dict(mjdata=pickle.dumps(base.data, protocol=5), wrappers=wrappers,
                rng=base.np_random.bit_generator.state,
                action_rng=vec.action_space.np_random.bit_generator.state,
                vector={k:getattr(vec,k) for k in ['buf_obs','buf_dones','buf_rews','buf_infos',
                                                 'reset_infos','_seeds','_options'] if hasattr(vec,k)})


def save(model, callbacks, path, commit):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    b = model.replay_buffer; n = b.buffer_size if b.full else b.pos
    keys = dict(model=model.key, policy=model.policy.key, noise=model.policy.noise_key,
                evaluation=model.policy.eval_key)
    aux = dict(commit=commit, model={k:getattr(model,k) for k in MODEL_FIELDS if hasattr(model,k)},
               callbacks=[{k:getattr(cb,k) for k in CALLBACK_FIELDS if hasattr(cb,k)} for cb in callbacks.callbacks],
               simulator=simulator(model.get_env()), numpy_rng=np.random.get_state(),
               python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
               keys={k:np.asarray(jax.random.key_data(v)) for k,v in keys.items()},
               typed_keys={k:str(v.dtype).startswith('key<') for k,v in keys.items()},
               replay=dict(n=n, pos=b.pos, full=b.full))
    temp = path.with_suffix('.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        z.writestr('states.msgpack', serialization.to_bytes(states(model)))
        z.writestr('aux.pkl', pickle.dumps(aux, protocol=5))
        for name in REPLAY_FIELDS:
            value = getattr(b,name,None)
            if value is not None:
                with z.open('replay/'+name+'.npy','w',force_zip64=True) as f:
                    np.lib.format.write_array(f,value[:n],allow_pickle=False)
    with temp.open('rb') as f: os.fsync(f.fileno())
    temp.replace(path)
    return dict(step=model.num_timesteps, replay_entries=n, bytes=path.stat().st_size)


def load(model, callbacks, path, commit):
    with zipfile.ZipFile(path) as z:
        aux = pickle.loads(z.read('aux.pkl')); assert aux['commit'] == commit
        st = serialization.from_bytes(states(model),z.read('states.msgpack'))
        model.policy.actor_state=st['explorer']; model.policy.target_actor_state=st['evaluator']
        model.policy.qf_state=st['critic']; model.ent_coef_state=st['entropy']
        for k,v in aux['model'].items(): setattr(model,k,v)
        for cb,row in zip(callbacks.callbacks,aux['callbacks']):
            for k,v in row.items(): setattr(cb,k,v)
        b=model.replay_buffer
        for name in REPLAY_FIELDS:
            filename='replay/'+name+'.npy'
            if filename in z.namelist():
                a=np.load(io.BytesIO(z.read(filename)),allow_pickle=False); getattr(b,name)[:len(a)]=a
        b.pos=aux['replay']['pos']; b.full=aux['replay']['full']
    vec=model.get_env(); vec.reset(); base=vec.envs[0].unwrapped; sim=aux['simulator']
    base.data=pickle.loads(sim['mjdata'])
    if getattr(base,'mujoco_renderer',None) is not None:base.mujoco_renderer.data=base.data
    base.np_random.bit_generator.state=sim['rng'];vec.action_space.np_random.bit_generator.state=sim['action_rng']
    env=vec.envs[0]
    for row in sim['wrappers']:
        env.__dict__.update(row)
        if hasattr(env,'env'):env=env.env
    for k,v in sim['vector'].items():setattr(vec,k,v)
    keys={k:jax.random.wrap_key_data(jnp.asarray(v)) if aux['typed_keys'][k] else jnp.asarray(v)
          for k,v in aux['keys'].items()}
    model.key=keys['model'];model.policy.key=keys['policy'];model.policy.noise_key=keys['noise']
    model.policy.eval_key=keys['evaluation']; model.policy.evaluation_only=False
    np.random.set_state(aux['numpy_rng']);random.setstate(aux['python_rng']);torch.set_rng_state(aux['torch_rng'])

