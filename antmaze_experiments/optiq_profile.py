"""Verify the requested AntMaze architecture and actual optimizer transforms."""
import json
from pathlib import Path
import numpy as np


def verify_profile(learner, folder):
    import jax.numpy as jnp
    policy = learner.model.policy
    cfg = learner.config['alg']
    assert list(policy.actor.hidden_dims) == [256, 256, 256]
    assert list(policy.qf.net_arch) == [256, 256, 256]
    assert cfg['optimizer']['lr_actor'] == 3e-4
    assert cfg['optimizer']['lr_critic'] == 5e-4
    assert cfg['tau'] == .005
    checks = {}
    # Isolated one-scalar optimizer states: do not update learner parameters,
    # counters, RNG, or the actual optimizer states.
    for label, state, expected in [('actor', policy.actor_state, 3e-4),
                                    ('critic', policy.qf_state, 5e-4)]:
        probe = {'probe': jnp.ones((1,), dtype=jnp.float32)}
        update, _ = state.tx.update(probe, state.tx.init(probe), probe)
        observed = -float(np.asarray(update['probe'])[0])
        assert np.isclose(observed, expected, rtol=1e-4, atol=1e-9), (label, observed)
        def shapes(tree, prefix=''):
            result = {}
            for key, value in tree.items():
                name = prefix + '/' + key
                if hasattr(value, 'items'):
                    result.update(shapes(value, name))
                elif key == 'kernel':
                    result[name] = list(value.shape)
            return result
        checks[label] = dict(expected_lr=expected, scalar_adam_probe=observed,
                             kernel_shapes=shapes(state.params))
    rnd_lrs = ([group['lr'] for group in learner.intrinsic.rnd_optimizer.param_groups]
               if learner.noveld_enabled else [])
    assert all(lr == 1e-4 for lr in rnd_lrs)
    result = dict(verified=True, actor_hidden_dims=[256, 256, 256],
                  critic_hidden_dims=[256, 256, 256], tau=.005,
                  rnd_lrs=rnd_lrs, optimizers=checks,
                  probe='Independent scratch optimizer states; learner state and RNG unchanged')
    (Path(folder)/'optiq-profile-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
