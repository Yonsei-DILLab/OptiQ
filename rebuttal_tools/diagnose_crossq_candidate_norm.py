"""Read-only GPU audit of archived MuJoCo checkpoints.

Original replay was not checkpointed. References below are explicitly labelled
fresh current-policy transitions or a mixture of historical-policy transitions.
No conclusion about the original replay distribution follows from this audit.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from omegaconf import OmegaConf

import environment
from ibolt.policy import IBOLTPolicy
from ibolt.box_gaussian import sample_box, mixture_log_prob
from rebuttal_tools.crossq_candidate_norm import reference_pass, pure_batch_reference, probe


@jax.jit
def score(actor, critic, obs, actions, next_obs, key, temperature):
    b, od = obs.shape
    n = 64
    ad = actor.params['mu']['bias'].shape[0]
    kz, ki, kn, ka = jax.random.split(key, 4)
    z = jax.random.normal(kz, (b*n, ad))
    repeated = jnp.broadcast_to(obs[:, None], (b, n, od)).reshape(b*n, od)
    (mu, ls), _ = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats},
        repeated, z, train=True, mutable=['batch_stats'])
    mu, ls = mu.reshape(b, n, ad), ls.reshape(b, n, ad)
    ids = jax.random.randint(ki, (b, n), 0, n)
    candidates = sample_box(kn, jnp.take_along_axis(mu, ids[..., None], 1),
                            jnp.take_along_axis(ls, ids[..., None], 1))
    density = mixture_log_prob(candidates, mu, ls)
    next_actions = IBOLTPolicy.sample_action(actor, next_obs, ka)
    _, refreshed, brn, ref = reference_pass(critic, critic.batch_stats, obs, actions, next_obs, next_actions)
    _, _, bn, _ = pure_batch_reference(critic, obs, actions, next_obs, next_actions)
    qs, ws = {}, {}
    for mode, stats in [('ema', critic.batch_stats), ('ema_refresh_probe', refreshed),
                        ('joint_reference', brn), ('joint_bn', bn)]:
        q = critic.apply_fn({'params': critic.params, 'batch_stats': stats},
                            repeated, candidates.reshape(b*n, ad), train=False).reshape(2, b, n).mean(0)
        qs[mode] = q
        ws[mode] = jax.nn.softmax(q/temperature-density, -1)
    return candidates, qs, ws


def collect(model, count, seed):
    import gymnasium as gym
    env = gym.make(model.cfg.env_name)
    obs, _ = env.reset(seed=seed)
    records, snapshots = [], []
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    key = jax.random.PRNGKey(seed)
    try:
        for i in range(count):
            physical = np.empty(mujoco.mj_stateSize(env.unwrapped.model, kind))
            mujoco.mj_getState(env.unwrapped.model, env.unwrapped.data, physical, kind)
            derived = {name: np.array(getattr(env.unwrapped.data, name))
                       for name in ('cinert', 'cvel', 'qfrc_actuator', 'cfrc_ext')}
            key, action_key = jax.random.split(key)
            a = np.asarray(IBOLTPolicy.sample_action(model.policy.actor_state, obs[None], action_key))[0]
            nxt, reward, term, trunc, _ = env.step(model.policy.unscale_action(a))
            records.append((np.array(obs), a, np.array(nxt)))
            # Gym's Humanoid/Ant observations include derived fields left by
            # mj_step. Preserve them as well as the integration state so the
            # initial observation is exactly the one scored by the actor.
            snapshots.append(dict(state=physical, derived=derived))
            obs = nxt
            if term or trunc:
                obs, _ = env.reset()
    finally:
        env.close()
    return tuple(np.stack([r[i] for r in records]) for i in range(3)), snapshots


@jax.jit
def paired_actions(actor, observations, keys):
    return jax.vmap(lambda ob, k: IBOLTPolicy.sample_action(actor, ob[None], k)[0])(observations, keys)


def mc_selected(model, states, observations, candidates, qs, ws, seed, horizon=1000):
    """Two common-random-number rollouts per selected candidate, six states.

    Measures the top-SNIS-weight candidate only, not the full weighted policy.
    Future actions use the frozen full stochastic policy without DACER noise,
    matching the critic's successor policy. Time limits are not terminal in TD.
    """
    import gymnasium as gym
    modes = list(ws)
    rows = []
    for state_id in range(min(6, len(states))):
        selected = {mode: int(ws[mode][state_id].argmax()) for mode in modes}
        selected['random_candidate'] = int(np.random.default_rng(seed+state_id).integers(64))
        unique = sorted(set(selected.values()))
        envs, candidate_ids, rep_ids = [], [], []
        try:
            for idx in unique:
                for rep in range(2):
                    env = gym.make(model.cfg.env_name).unwrapped
                    env.reset(seed=seed+state_id*10+rep)
                    mujoco.mj_setState(env.model, env.data, states[state_id]['state'], mujoco.mjtState.mjSTATE_INTEGRATION)
                    mujoco.mj_forward(env.model, env.data)
                    for name, value in states[state_id]['derived'].items():
                        getattr(env.data, name)[:] = value
                    np.testing.assert_allclose(env._get_obs(), observations[state_id], rtol=2e-5, atol=2e-5)
                    envs.append(env); candidate_ids.append(idx); rep_ids.append(rep)
            current = np.repeat(observations[state_id][None], len(envs), axis=0)
            returns = np.zeros(len(envs)); active = np.ones(len(envs), dtype=bool)
            lengths = np.zeros(len(envs), dtype=int)
            for t in range(horizon):
                if t == 0:
                    actions = np.stack([candidates[state_id, i] for i in candidate_ids])
                else:
                    # Same latent and conditional-noise draw across alternatives.
                    padded_reps = rep_ids + [0]*(10-len(rep_ids))
                    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(seed+state_id*10+r), t)
                                      for r in padded_reps])
                    padded_obs = np.pad(current, ((0,10-len(envs)), (0,0)))
                    actions = np.asarray(paired_actions(model.policy.actor_state, padded_obs, keys))[:len(envs)]
                for i, env in enumerate(envs):
                    if not active[i]:
                        continue
                    nxt, reward, term, _, _ = env.step(model.policy.unscale_action(actions[i]))
                    returns[i] += float(model.gamma)**t*reward
                    lengths[i] += 1
                    current[i] = nxt
                    active[i] = not term
                if not active.any():
                    break
            by_candidate = {str(i): returns[np.asarray(candidate_ids)==i].tolist() for i in unique}
            values = {m: float(np.mean(by_candidate[str(i)])) for m, i in selected.items()}
            rows.append(dict(state=state_id, selected=selected, returns=values,
                candidate_returns=by_candidate, lengths=lengths.tolist(),
                predicted_q={m: float(qs[m][state_id, selected[m]]) for m in modes}))
        finally:
            for env in envs:
                env.close()
    return dict(horizon=horizon, replicates=2, states=len(rows),
        metric='Discounted rollout return of top-SNIS-weight candidate, not full weighted policy',
        rows=rows, mean={m: float(np.mean([r['returns'][m] for r in rows])) for m in [*modes, 'random_candidate']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mc', action='store_true')
    parser.add_argument('--steps', type=int, nargs='+')
    args = parser.parse_args()
    assert jax.default_backend() == 'gpu'
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.run/'config.json').read_text())
    cfg = OmegaConf.create({k:v for k,v in original.items() if k not in ('runtime','protocol')})
    cfg.output_root = str(args.output/'model-shell')
    cfg.alg.buffer_size = 1024
    cfg.wandb.activate = False
    model, callbacks = environment.create_algorithm(cfg)
    checkpoint_dir = next((args.run/'checkpoints').glob('*'))
    steps = sorted(int(f.stem.rsplit('_',1)[1]) for f in checkpoint_dir.glob('actor_state_*.msgpack'))
    selected_steps = sorted(set(args.steps or [100000, 400000, max(steps)]) & set(steps))
    records = []
    started = time.time()
    try:
        model.load_model(checkpoint_dir, min(steps), min(steps))
        old, _ = collect(model, 1024, 722)
        for step in selected_steps:
            model.load_model(checkpoint_dir, step, step)
            current, states = collect(model, 1024, 733+step)
            subsets = dict(current=current,
                           historical_current_mix=tuple(np.concatenate([a,b]) for a,b in zip(old,current)))
            row = dict(step=step, references={})
            for label, dataset in subsets.items():
                rng = np.random.default_rng(744+step)
                indices = rng.choice(len(dataset[0]), 256, replace=False)
                batch = [jnp.asarray(a[indices]) for a in dataset]
                metrics = probe(model.policy.actor_state, model.policy.qf_state, *batch,
                    jax.random.PRNGKey(755+step), float(cfg.alg.actor.temperature), 1.)
                row['references'][label] = {k:float(v) for k,v in metrics.items()}
            # The first six candidates correspond to simulator states retained above.
            chosen = np.arange(256)
            candidates, qs, ws = score(model.policy.actor_state, model.policy.qf_state,
                *[jnp.asarray(a[chosen]) for a in current], jax.random.PRNGKey(766+step),
                float(cfg.alg.actor.temperature))
            candidates = np.asarray(candidates)
            qs, ws = ({k:np.asarray(v) for k,v in d.items()} for d in (qs, ws))
            # Spread selected physical states out across the trajectory.
            mc_ids = np.array([16, 48, 80, 112, 176, 240])
            if args.mc and step == max(selected_steps):
                row['mc'] = mc_selected(model, [states[i] for i in mc_ids], current[0][mc_ids],
                    candidates[mc_ids], {k:v[mc_ids] for k,v in qs.items()},
                    {k:v[mc_ids] for k,v in ws.items()}, seed=777+step)
            records.append(row)
            result = dict(task=cfg.task, seed=int(cfg.seed), source_run=str(args.run),
                backend=jax.default_backend(), devices=[d.device_kind for d in jax.devices()],
                temperature=float(cfg.alg.actor.temperature), elapsed_seconds=time.time()-started,
                limitation='Original replay unavailable. Fresh current and historical-policy transitions are surrogate reference batches.',
                records=records)
            tmp=args.output/'result.tmp.json';tmp.write_text(json.dumps(result,indent=2));tmp.replace(args.output/'result.json')
            print(json.dumps(dict(task=cfg.task,step=step,elapsed=result['elapsed_seconds'],
                current=row['references']['current'],mc=row.get('mc',{}).get('mean'))),flush=True)
    finally:
        model.get_env().close();model.logger.close()
        for cb in callbacks.callbacks:
            if hasattr(cb,'eval_env'):cb.eval_env.close()


if __name__ == '__main__':
    main()
