import inspect
import math
from pathlib import Path

import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy.stats import norm
from stable_baselines3.common.logger import configure

from optiq_dime import OptiQDIME
from optiq_dime.schedules import proposal_parameters
from optiq_dime.transport import TruncatedGaussianKDE
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(name="optiq_dime_reach_proposal_anneal", overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=name, overrides=list(overrides))


def test_comparison_changes_only_proposal_schedule_and_labels():
    baseline = OmegaConf.to_container(config("optiq_dime_no_anchor", ["benchmark=reach_hard"]), resolve=True)
    cfg = config()
    assert validate_config(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    assert actual["alg"]["actor"].pop("proposal_schedule") == {
        "initial_temperature": 20.0, "hold_steps": 50000, "end_steps": 200000}
    for key in ("output_root", "run_name"):
        actual.pop(key); baseline.pop(key)
    actual["wandb"].pop("group"); baseline["wandb"].pop("group")
    assert actual == baseline


@pytest.mark.parametrize("step,factor", [(0,20), (5000,20), (50000,20), (125000,10.5), (200000,1), (1000000,1)])
def test_schedule_boundaries(step, factor):
    cfg = config()
    std, clip, value = proposal_parameters(cfg.alg.actor, step)
    assert value == factor
    assert std == pytest.approx(0.2 * math.sqrt(factor))
    assert clip == pytest.approx(0.5 * math.sqrt(factor))


@pytest.mark.parametrize("override", ["initial_temperature=0", "initial_temperature=nan", "hold_steps=-1", "end_steps=50000", "end_steps=1000001"])
def test_invalid_schedule_rejected(override):
    with pytest.raises(ValueError, match="proposal_schedule"):
        validate_config(config(overrides=["alg.actor.proposal_schedule." + override]))


def test_wide_kernel_sampling_and_density_agree_with_truncated_normal():
    cfg = config()
    std, clip, _ = proposal_parameters(cfg.alg.actor, 5000)
    kde = TruncatedGaussianKDE.from_centers(jnp.zeros((1024,16,1)), std, clip)
    samples = np.asarray(kde.sample_stratified(jax.random.PRNGKey(31),4)).reshape(1024,64,1)
    assert np.max(np.abs(samples)) <= 1
    normalizer = norm.cdf(1/std) - norm.cdf(-1/std)
    tail_probability = 1 - (norm.cdf(0.5/std) - norm.cdf(-0.5/std)) / normalizer
    assert np.mean(np.abs(samples) > 0.5) == pytest.approx(tail_probability, abs=0.01)
    expected = norm.logpdf(samples[:,:,0], scale=std) - math.log(normalizer)
    np.testing.assert_allclose(kde.log_prob(jnp.asarray(samples)), expected, atol=1e-6)
    baseline = TruncatedGaussianKDE.from_centers(kde.centers,0.2,0.5)
    assert np.max(np.abs(baseline.sample_stratified(jax.random.PRNGKey(31),4))) <= 0.500001


def test_training_anneals_only_candidates_and_keeps_td_and_q_temperature():
    cfg = config(overrides=["benchmark=ant", "alg.critic.hs=[32,32]", "alg.actor.hidden_dims=[32,32]",
        "alg.buffer_size=32", "alg.batch_size=4", "alg.learning_starts=2", "alg.actor.learning_starts=2",
        "alg.actor.proposal_schedule.hold_steps=3", "alg.actor.proposal_schedule.end_steps=6"])
    model = OptiQDIME("MlpPolicy",gym.make("Ant-v4"),None,1,cfg)
    model.set_logger(configure(None,[]))
    captured=[]
    original=model._train
    signature=inspect.signature(original)
    def tracked(*args, **kwargs):
        bound=signature.bind(*args,**kwargs).arguments
        captured.append({k:bound[k] for k in ("proposal_std","proposal_clip","temperature","td_noise_std","td_noise_clip")})
        return original(*args,**kwargs)
    model._train=tracked
    try:
        model.learn(total_timesteps=6)
        assert len(captured)==4
        for step, record in zip(range(3,7),captured):
            std,clip,_=proposal_parameters(cfg.alg.actor,step)
            assert record=={"proposal_std":std,"proposal_clip":clip,"temperature":0.25,"td_noise_std":0.2,"td_noise_clip":0.5}
        assert model._n_updates==8
        assert model.logger.name_to_value["train/proposal_temperature"]==1
        assert np.isfinite(model.logger.name_to_value["train/actor_loss"])
    finally:
        model.get_env().close()
