"""GPU checks for the candidate-normalization intervention, before any training."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from models.critic import CrossQVectorCritic
from models.native_critic import CrossQVectorCritic as NativeCritic
from ibolt.algorithm import IBOLT
from rebuttal_tools.crossq_candidate_norm import NATIVE_STEP, make_step, reference_pass


def maximum_difference(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return max(float(jnp.max(jnp.abs(x-y))) for x, y in zip(la, lb) if jnp.size(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == 'gpu', 'GPU required'
    key = jax.random.PRNGKey(123)
    x = jax.random.normal(key, (64, 11))
    a = jax.random.uniform(key, (64, 3), minval=-1, maxval=1)
    checks = {}
    for warmup in (0, 100000):
        q = CrossQVectorCritic(net_arch=(32, 32), warmup_steps=warmup)
        native = NativeCritic(net_arch=(32, 32), warmup_steps=warmup)
        v = q.init(key, x, a, train=False)
        original = native.init(key, x, a, train=False)
        assert maximum_difference(v, original) == 0
        # Deliberately exercise d-clipping in addition to normal warmup.
        flat = flatten_dict(unfreeze(v['batch_stats']))
        v['batch_stats'] = unflatten_dict({k: (jnp.ones_like(val)*9 if k[-1]=='mean' else val)
                                           for k, val in flat.items()})
        y, u = q.apply(v, x, a, train=True, mutable=['batch_stats', 'reference_stats'])
        yn, un = native.apply(v, x, a, train=True, mutable=['batch_stats'])
        assert maximum_difference((y, u['batch_stats']), (yn, un['batch_stats'])) == 0
        effective = unflatten_dict({k: value for k, value in flatten_dict(unfreeze(u['reference_stats'])).items()
                                    if k[-1] in ('mean', 'var', 'steps')})
        fixed = q.apply({'params': v['params'], 'batch_stats': effective}, x, a, train=False)
        error = float(jnp.max(jnp.abs(y-fixed)))
        assert error < 2e-5, error
        split = jnp.concatenate([q.apply({'params': v['params'], 'batch_stats': effective},
                    x[:31], a[:31], train=False),
                    q.apply({'params': v['params'], 'batch_stats': effective}, x[31:], a[31:], train=False)], axis=1)
        split_error = float(jnp.max(jnp.abs(split-fixed)))
        assert np.allclose(np.asarray(split), np.asarray(fixed), atol=2e-5, rtol=2e-5), split_error
        checks[f'split_batch_max_abs_error_warmup_{warmup}'] = split_error
        checks[f'fixed_reference_error_warmup_{warmup}'] = error
        checks[f'capture_preserves_native_warmup_{warmup}'] = True
    import train
    import environment
    cfg = train.compose_config(['benchmark=hopper', '+critic=crossq', '+actor=crossq',
        'alg.critic.hs=[32,32]', 'alg.actor.hidden_dims=[32,32]', 'wandb.activate=false',
        'dacer.enabled=false', 'alg.buffer_size=1024',
        f'output_root={args.output.parent}/check-environment'])
    model, callbacks = environment.create_algorithm(cfg)
    try:
        actor, critic = model.policy.actor_state, model.policy.qf_state
        arguments = (0.99, 0.005, actor, critic, x, a, x+.2,
                     jnp.zeros(64), jnp.ones(64), key, 8, 1, .5, 1., True)
        original = NATIVE_STEP(*arguments, crossq=True, use_min_candidate_q=False)
        control = make_step('ema')(IBOLT, *arguments, crossq=True, use_min_candidate_q=False)
        # The original state and random stream are bit-identical in the control.
        assert maximum_difference(original, control) == 0
        checks['ema_control_identical_to_native_step'] = True
        for mode in ('ema_postupdate', 'joint_reference', 'joint_bn'):
            result = make_step(mode)(IBOLT, *arguments, crossq=True, use_min_candidate_q=False)
            assert maximum_difference(result[0], original[0]) == 0
            assert maximum_difference(result[-1], original[-1]) == 0
            assert maximum_difference(result[1].batch_stats, original[1].batch_stats) == 0
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(result)
                       if hasattr(v, 'dtype') and not jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key))
            checks[mode+'_critic_rng_actor_stats_unchanged'] = True
    finally:
        model.get_env().close()
        model.logger.close()
        for cb in callbacks.callbacks:
            if hasattr(cb, 'eval_env'):
                cb.eval_env.close()
    args.output.write_text(json.dumps(dict(backend=jax.default_backend(),
        devices=[d.device_kind for d in jax.devices()], checks=checks), indent=2))
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
