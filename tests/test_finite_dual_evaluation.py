"""Evaluation must sample the trained finite support, with no Gaussian epsilon."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra import compose, initialize_config_dir
from pathlib import Path
from optiq_dime.latent import FiniteMixtureTrainState, finite_latent_codes
from optiq_dime.policy import OptiQPolicy
from optiq_dime.dual_evaluation import dual_mu_evaluation_spec
from run_optiq_dime import validate_config


def test_stochastic_finite_evaluation_uses_training_codes():
    # An identity mu makes any out-of-codebook evaluation action observable.
    # A huge sigma also makes accidental conditional noise easy to detect.
    def apply_fn(params, observations, z):
        return z, jnp.full_like(z, 10.)
    state = FiniteMixtureTrainState.create(apply_fn=apply_fn,
        params={"mu":{"bias":jnp.zeros(17)}}, tx=optax.adam(3e-4),
        latent_components=64, latent_codebook_seed=20260911)
    obs=jnp.zeros((4096, 376));key=jax.random.PRNGKey(41)
    action=np.asarray(OptiQPolicy.sample_action(state,obs,key,False,False))
    support=np.asarray(jnp.tanh(finite_latent_codes(state,17)))
    distance=np.max(np.abs(action[:,None,:]-support[None,:,:]),axis=-1)
    assert distance.min(axis=1).max()<1e-6
    counts=np.bincount(distance.argmin(axis=1),minlength=64)
    assert counts.min()>25 and counts.max()<110
    other=np.asarray(OptiQPolicy.sample_action(state,obs,jax.random.PRNGKey(42),False,False))
    assert not np.array_equal(action,other)
    fixed=np.asarray(OptiQPolicy.sample_action(state,obs,key,True,False))
    np.testing.assert_allclose(fixed,np.broadcast_to(support[0],fixed.shape),atol=1e-6)


def test_finite_and_continuous_evaluation_specs():
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        finite=compose(config_name='mujoco_v5_direct_gmm_humanoid_fixed64')
        continuous=compose(config_name='mujoco_v5_direct_gmm',overrides=['benchmark=humanoid'])
    validate_config(finite);validate_config(continuous)
    mode,spec=dual_mu_evaluation_spec(finite)
    assert mode=='fixed_z' and spec['legacy_eval_alias']=='stochastic_z'
    assert spec['latent_components']==64 and spec['latent_codebook_seed']==20260911
    assert 'same fixed training codebook' in spec['stochastic_z']
    mode,spec=dual_mu_evaluation_spec(continuous)
    assert mode=='zero_z' and spec['legacy_eval_alias']=='zero_z'
    assert 'z~N(0,I)' in spec['stochastic_z']
