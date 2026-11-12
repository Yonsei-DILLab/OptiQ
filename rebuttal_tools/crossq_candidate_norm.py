"""Isolated CrossQ candidate-normalization experiment; no default learner changes.

The source builder adds an opt-in reference_stats collection to the native BRN.
All three arms perform exactly the native critic update. Only the temporary
critic state used for the actor's candidate Q evaluations differs.
"""
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from stable_baselines3.common.callbacks import BaseCallback

from ibolt.algorithm import IBOLT
from ibolt.policy import IBOLTPolicy
from ibolt.box_gaussian import sample_box, mixture_log_prob

MODES = ('ema', 'ema_postupdate', 'joint_reference', 'joint_bn')
NATIVE_STEP = IBOLT._step


def reference_pass(critic, previous_stats, obs, actions, next_obs, next_actions):
    """Recompute with current weights, starting from the pre-update EMA once.

    This is read-only with respect to the live critic. In particular, neither
    the EMA nor its warmup counter is advanced a second time in the learner.
    """
    q, updates = critic.apply_fn(
        {'params': critic.params, 'batch_stats': previous_stats},
        jnp.concatenate([obs, next_obs]), jnp.concatenate([actions, next_actions]),
        train=True, mutable=['batch_stats', 'reference_stats'])
    reference = flatten_dict(unfreeze(updates['reference_stats']))
    effective = unflatten_dict({k: v for k, v in reference.items()
                               if k[-1] in ('mean', 'var', 'steps')})
    return q, updates['batch_stats'], effective, reference


def pure_batch_reference(critic, obs, actions, next_obs, next_actions):
    """Raw BN at every layer, independent of EMA, in a read-only reference pass.

    A temporary pre-warmup counter disables r/d correction. Its statistics are
    never written into the live critic. Layers see the preceding raw-BN layer,
    rather than using raw moments collected under a different BRN transform.
    """
    flat = flatten_dict(unfreeze(critic.batch_stats))
    temporary = unflatten_dict({k: -jnp.ones_like(v) if k[-1]=='steps' else v
                                for k, v in flat.items()})
    return reference_pass(critic, temporary, obs, actions, next_obs, next_actions)


def make_step(mode):
    if mode not in MODES:
        raise ValueError(mode)

    @partial(jax.jit, static_argnames=['cls', 'n', 'repeats', 'do_actor',
                                     'crossq', 'use_min_candidate_q'])
    def step(cls, gamma, tau, actor, critic, obs, actions, next_obs, dones,
             rewards, key, n, repeats, temperature, beta, do_actor,
             crossq=False, use_min_candidate_q=False):
        assert crossq, 'This experiment is specific to the CrossQ critic.'
        before = critic
        key_before = key
        critic, metrics, key = cls.update_critic_crossq(
            gamma, actor, critic, obs, actions, next_obs, dones, rewards, key)
        if do_actor:
            scorer = critic
            if mode != 'ema':
                # Reproduce the exact successor actions already used by TD.
                _, actor_key, _, _, _, _ = jax.random.split(key_before, 6)
                next_actions = jax.lax.stop_gradient(
                    IBOLTPolicy.sample_action(actor, next_obs, actor_key))
                if mode == 'joint_bn':
                    _, _, stats, _ = pure_batch_reference(
                        critic, obs, actions, next_obs, next_actions)
                else:
                    _, refreshed, effective, _ = reference_pass(
                        critic, before.batch_stats, obs, actions, next_obs, next_actions)
                    stats = refreshed if mode == 'ema_postupdate' else effective
                scorer = critic.replace(batch_stats=stats)
            actor, actor_metrics, key = cls.update_actor(
                actor, scorer, obs, key, n, repeats, temperature, beta,
                use_min_candidate_q=use_min_candidate_q)
            metrics.update(actor_metrics)
        return critic, actor, metrics, key
    return step


def install(mode):
    IBOLT._step = classmethod(make_step(mode))


@partial(jax.jit, static_argnames=['n', 'repeats'])
def probe(actor, critic, obs, actions, next_obs, key, temperature, beta, n=64, repeats=1):
    """Frozen same-candidate comparison. No training RNG/state is modified.

    This diagnostic starts from the current stored EMA, without an optimizer
    step, so ema_postupdate here measures an additional one-batch refresh.
    The paired one-update self-test separately tests its exact training path.
    """
    key_z, key_a, key_i, key_noise = jax.random.split(key, 4)
    b, od = obs.shape
    ad = actor.params['mu']['bias'].shape[0]
    z = jax.random.normal(key_z, (b*n, ad))
    repeated = jnp.broadcast_to(obs[:, None], (b, n, od)).reshape(b*n, od)
    (mu, ls), _ = actor.apply_fn(
        {'params': actor.params, 'batch_stats': actor.batch_stats}, repeated, z,
        train=True, mutable=['batch_stats'])
    mu, ls = mu.reshape(b, n, ad), ls.reshape(b, n, ad)
    m = n*repeats
    ids = jax.random.randint(key_i, (b, m), 0, n)
    candidates = sample_box(key_noise, jnp.take_along_axis(mu, ids[..., None], 1),
                            jnp.take_along_axis(ls, ids[..., None], 1))
    density = mixture_log_prob(candidates, mu, ls)
    next_actions = IBOLTPolicy.sample_action(actor, next_obs, key_a)
    _, refreshed, effective, ref = reference_pass(
        critic, critic.batch_stats, obs, actions, next_obs, next_actions)
    _, _, raw_bn, _ = pure_batch_reference(critic, obs, actions, next_obs, next_actions)
    candidate_obs = jnp.broadcast_to(obs[:, None], (b, m, od)).reshape(b*m, od)
    result = {}
    base_q = base_w = None
    for mode, stats in [('ema', critic.batch_stats), ('ema_refresh_probe', refreshed),
                        ('joint_reference', effective), ('joint_bn', raw_bn)]:
        q = critic.apply_fn({'params': critic.params, 'batch_stats': stats},
                            candidate_obs, candidates.reshape(b*m, ad), train=False)
        q = q.reshape(2, b, m).mean(0)
        w = jax.nn.softmax(q/temperature - beta*density, axis=-1)
        if base_q is None:
            base_q, base_w = q, w
        centered_delta = (q-base_q) - (q-base_q).mean(-1, keepdims=True)
        result[mode+'/ess_mean'] = jnp.mean(1/jnp.sum(w*w, -1))
        result[mode+'/max_weight_mean'] = w.max(-1).mean()
        result[mode+'/q_span_mean'] = jnp.ptp(q, axis=-1).mean()
        result[mode+'/centered_q_delta_rms_over_T'] = jnp.sqrt(jnp.mean(centered_delta**2))/temperature
        result[mode+'/weight_tv_mean'] = (jnp.abs(w-base_w).sum(-1)/2).mean()
        result[mode+'/top_q_change_fraction'] = (q.argmax(-1) != base_q.argmax(-1)).mean()
    for key, val in ref.items():
        if key[-1] in ('r_clip_fraction', 'd_clip_fraction', 'warm', 'steps'):
            result['brn/'+'/'.join(key)] = val.mean()
    return result


class NormProbeCallback(BaseCallback):
    def __init__(self, every=25_000):
        super().__init__()
        self.every = every

    def _on_step(self):
        if self.num_timesteps % self.every or self.model.replay_buffer.size() < 256:
            return True
        model = self.model
        saved = np.random.get_state()
        try:
            np.random.seed(int(model.seed)*1_000_003 + self.num_timesteps)
            data = model.replay_buffer.sample(256)
        finally:
            np.random.set_state(saved)
        a = model.cfg.alg.actor
        metrics = probe(model.policy.actor_state, model.policy.qf_state,
            jnp.asarray(data.observations.numpy()), jnp.asarray(data.actions.numpy()),
            jnp.asarray(data.next_observations.numpy()),
            jax.random.PRNGKey(int(model.seed)*1_000_003 + self.num_timesteps),
            float(a.temperature), float(a.density_correction_beta),
            int(a.num_policy_samples), int(a.proposals_per_policy_sample))
        values = {k: float(v) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError('Nonfinite normalization diagnostics')
        row = dict(env_steps=self.num_timesteps, metrics=values)
        with (Path(model.cfg.output_root)/'candidate_norm.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        for k, v in values.items():
            model.logger.record('candidate_norm/'+k, v)
        return True


def main():
    import argparse
    import environment
    import train
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Server GPU is mandatory for this experiment.')
    install(args.mode)
    native_create = environment.create_algorithm
    def create(cfg):
        model, callbacks = native_create(cfg)
        callbacks.callbacks.append(NormProbeCallback())
        # Keep CSV and TensorBoard; episode-level stdout tables are redundant.
        model.logger.output_formats = [f for f in model.logger.output_formats
                                       if f.__class__.__name__ != 'HumanOutputFormat']
        return model, callbacks
    environment.create_algorithm = create
    cfg = train.compose_config(args.overrides)
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, 'candidate_normalization', args.mode, force_add=True)
    train.run(cfg)


if __name__ == '__main__':
    main()
