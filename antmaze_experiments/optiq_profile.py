"""Verify the selected OptiQ AntMaze architecture and optimizer transforms."""
import json
from pathlib import Path
import numpy as np


def verify_profile(learner, folder):
    import jax.numpy as jnp
    policy = learner.model.policy
    cfg = learner.config['alg']
    config_profile = learner.optiq_config_profile
    basic = config_profile == 'basic'
    expected_hidden = [256, 256] if basic else [256, 256, 256]
    dynamics = getattr(learner, 'dynamics_profile', None)
    if dynamics is not None:
        from .dynamics_profiles import get_profile
        expected_profile = get_profile(dynamics)
    else:
        expected_profile = dict(critic_lr=3e-4 if basic else 5e-4, tau=.005, policy_delay=1)
    assert list(cfg['actor']['hidden_dims']) == expected_hidden
    assert list(cfg['critic']['hs']) == expected_hidden
    assert cfg['actor']['mean_output_init_scale'] == (1e-4 if basic else 1.)
    assert cfg['optimizer']['lr_actor'] == 3e-4
    assert cfg['optimizer']['lr_critic'] == expected_profile['critic_lr']
    assert cfg['tau'] == expected_profile['tau']
    if dynamics is not None:
        assert cfg['policy_delay'] == expected_profile['policy_delay']
    checks = {}
    # Isolated one-scalar optimizer states: do not update learner parameters,
    # counters, RNG, or the actual optimizer states.
    for label, state, expected in [('actor', policy.actor_state, 3e-4),
                                    ('critic', policy.qf_state, expected_profile['critic_lr'])]:
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
        kernels = shapes(state.params)
        hidden = [shape for shape in kernels.values() if shape[-1] == 256]
        expected_kernels = ([[37,256]] + [[256,256]] * (len(expected_hidden)-1)
                            if label == 'actor' else
                            [[2,37,256]] + [[2,256,256]] * (len(expected_hidden)-1))
        assert sorted(hidden) == sorted(expected_kernels), (label, kernels)
        checks[label] = dict(expected_lr=expected, scalar_adam_probe=observed,
                             kernel_shapes=kernels)
    rnd_lrs = ([group['lr'] for group in learner.intrinsic.rnd_optimizer.param_groups]
               if learner.noveld_enabled else [])
    assert all(lr == 1e-4 for lr in rnd_lrs)
    result = dict(verified=True, config_profile=config_profile,
                  actor_hidden_dims=expected_hidden,
                  critic_hidden_dims=expected_hidden, tau=cfg['tau'],
                  policy_delay=cfg['policy_delay'], dynamics_profile=dynamics,
                  rnd_lrs=rnd_lrs, optimizers=checks,
                  probe='Independent scratch optimizer states; learner state and RNG unchanged')
    (Path(folder)/'optiq-profile-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
