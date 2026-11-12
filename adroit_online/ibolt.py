"""Connect the unchanged repository iBOLT/DACER learner to online-only replay."""
import copy
import pickle
from pathlib import Path

from .run import atomic


def create_model(c, env, output):
    import jax
    from omegaconf import OmegaConf
    from exploration import ExplorationIBOLT
    from train import compose_config, validate_config

    assert output is not None and jax.default_backend() == 'gpu'
    cfg = compose_config([
        f"env_name={c['env_id']}", f"task=adroit-{c['task']}", f"seed={c['seed']}",
        f"temperature={c['temperature']}", f"total_steps={c['steps']}",
        f"alg.learning_starts={c['warmup']}", f"alg.actor.learning_starts={c['warmup']}",
        f"alg.utd={c['utd']}", f"alg.buffer_size={c['replay_capacity']}",
        f"alg.actor.hidden_dims={c['hidden']}", f"alg.critic.hs={c['hidden']}",
        f"alg.batch_size={c['batch_size']}", f"alg.gamma={c['gamma']}", f"alg.tau={c['tau']}",
        f"alg.optimizer.lr_actor={c['learning_rate']}", f"alg.optimizer.lr_critic={c['learning_rate']}",
        'wandb.activate=false', f'output_root={output}', 'diagnostic_interval=0',
        f"checkpoint_interval={c['checkpoint_every']}"])
    validate_config(cfg)
    assert cfg.alg.actor.num_policy_samples == 64 and cfg.alg.actor.proposals_per_policy_sample == 1
    assert cfg.alg.actor.log_std_min == -5 and cfg.alg.actor.log_std_max == -1
    assert cfg.alg.actor.initial_log_std == -1 and cfg.alg.actor.density_correction_beta == 1
    assert not cfg.alg.actor.use_min_candidate_q and cfg.alg.actor.policy_delay == 1
    assert cfg.alg.critic.type == 'default' and not cfg.alg.actor.batch_renorm
    assert cfg.dacer.enabled and cfg.dacer.noise_scale == .1
    assert cfg.dacer.target_entropy_per_dim == -.9 and cfg.dacer.initial_alpha == .27
    assert cfg.dacer.alpha_lr == .03 and cfg.dacer.interval_updates == 10000
    atomic(Path(output) / 'ibolt-config.json', OmegaConf.to_container(cfg, resolve=True))
    # An empty model_save_path disables the learner's independent autosave
    # schedule. The common runner invokes the native serializer at paired steps.
    return ExplorationIBOLT('MlpPolicy', env=env, cfg=cfg, model_save_path='',
                            save_every_n_steps=c['checkpoint_every'],
                            gradient_steps=c['gradient_steps'])


def rng_state(model):
    import jax
    import numpy as np
    return dict(learner_key=np.asarray(jax.random.key_data(model.key)).copy(),
                policy_key=np.asarray(jax.random.key_data(model.policy.key)).copy(),
                policy_noise_key=np.asarray(jax.random.key_data(model.policy.noise_key)).copy(),
                regulator_key=np.asarray(jax.random.key_data(model.regulator_key)).copy(),
                regulator_numpy=copy.deepcopy(model.regulator_rng.bit_generator.state),
                evaluation_mu_only=bool(getattr(model.policy, 'evaluation_mu_only', False)))


def checkpoint(model, folder, verify=False):
    import flax.serialization
    import jax
    import numpy as np
    folder = Path(folder)
    states = (model.policy.actor_state, model.policy.qf_state)
    for state in states:
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    previous = model.model_save_path
    try:
        model.model_save_path = str(folder)
        model._save_model()
    finally:
        model.model_save_path = previous
    auxiliary = dict(rng_state(model), steps=model.num_timesteps, updates=model._n_updates,
                     regulator_log_alpha=np.asarray(model.regulator_log_alpha),
                     regulator_optimizer=jax.tree_util.tree_map(np.asarray, model.regulator_state),
                     regulator_next_update=model.regulator_next_update,
                     regulator_entropy=model.regulator_entropy)
    (folder / f'auxiliary-state-{model.num_timesteps:09d}.pkl').write_bytes(pickle.dumps(auxiliary))
    if verify:
        for name, state in zip(('actor_state', 'critic_state'), states):
            restored = flax.serialization.from_bytes(
                state, (folder / f'{name}_{model.num_timesteps}.msgpack').read_bytes())
            for a, b in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(restored), strict=True):
                assert np.array_equal(np.asarray(a), np.asarray(b)), 'Checkpoint changed a parameter/optimizer leaf'
        atomic(folder / 'restore-validation.json', dict(passed=True, steps=model.num_timesteps,
                                                       actor_and_critic_optimizer_exact=True))
