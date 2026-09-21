"""Check resolved profiles and actual actor parameters without training."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from train_mean1 import compose_config
from optiq_dime.policy import SemiImplicitActor

report = []
for task, obs_dim, action_dim, seeds in (("halfcheetah", 17, 6, range(2)), ("ant", 27, 8, range(4))):
    for seed in seeds:
        cfg = compose_config(task, seed, "/tmp/optiq-mean1-profile-validation")
        assert cfg.alg.actor.mean_output_init_scale == 1.
    obs = jnp.zeros((64, obs_dim))
    z = jax.random.normal(jax.random.PRNGKey(33), (64, action_dim))
    actors = [SemiImplicitActor(action_dim, (256, 256), -5., -1., -1.,
                               mean_output_init_scale=scale) for scale in (1e-4, 1.)]
    params = [actor.init(jax.random.PRNGKey(7), obs, z) for actor in actors]
    flat = [flatten_dict(p) for p in params]
    assert flat[0].keys() == flat[1].keys()
    for path in flat[0]:
        old, new = np.asarray(flat[0][path]), np.asarray(flat[1][path])
        if path[-2:] == ("mu", "kernel"):
            np.testing.assert_allclose(new, 100 * old, rtol=2e-6, atol=1e-8)
        else:
            np.testing.assert_array_equal(new, old)
    mu, log_std = actors[1].apply(params[1], obs, z)
    assert np.isfinite(mu).all() and (np.abs(mu) <= 1).all()
    np.testing.assert_array_equal(log_std, -np.ones_like(log_std))
    report.append(dict(task=task, seeds=list(seeds), mean_kernel_std_ratio=100,
                       other_initial_parameters="identical", log_std_min=-5,
                       log_std_max=-1, initial_log_std=-1, dacer_noise_scale=cfg.dacer.noise_scale))
print(json.dumps(dict(status="passed", profiles=report), indent=2))
