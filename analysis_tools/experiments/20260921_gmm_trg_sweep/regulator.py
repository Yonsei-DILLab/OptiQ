"""DACER Eq15/16 behavior-only adaptation; not a change to teacher or Bellman target.

Paper: https://arxiv.org/html/2405.15177v4 (Table3).
Audited official code: happy-yan/DACER-Diffusion-with-Online-RL,
9f22f29fa91b1bed8ee07177a51598d6bd413cb5, relax/algorithm/dacer.py.
The fitted-GMM joint entropy is a proxy (upper bound on fitted mixture entropy),
not exact entropy, and clipping produces boundary atoms. Keep this distinction.
"""
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.policy import OptiQPolicy

def entropy_proxy(actions, components=3, random_state=42):
    from sklearn.mixture import GaussianMixture
    actions = np.asarray(actions, dtype=np.float64)
    assert actions.ndim == 3 and np.isfinite(actions).all()
    values = []
    converged = []
    for samples in actions:
        gmm = GaussianMixture(n_components=components, covariance_type='full',
                              random_state=random_state).fit(samples)
        sign, logdet = np.linalg.slogdet(gmm.covariances_)
        assert np.all(sign > 0)
        component_h = 0.5 * (samples.shape[-1] * (1 + np.log(2*np.pi)) + logdet)
        w = gmm.weights_
        values.append(float(-np.dot(w, np.log(w)) + np.dot(w, component_h)))
        converged.append(gmm.converged_)
    result = float(np.mean(values))
    assert np.isfinite(result)
    return result, float(np.mean(converged))

def noisy_action(action, noise, noise_std):
    return np.clip(action + noise_std * noise, -1., 1.).astype(action.dtype)

class BehaviorRegulatedOptiQ(OptiQDIME):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regulator_cfg = self.cfg.dacer
        self.regulator_enabled = bool(self.regulator_cfg.enabled)
        if not self.regulator_enabled: return
        c = self.regulator_cfg
        self.regulator_log_alpha = jnp.array(np.log(c.initial_alpha), dtype=jnp.float32)
        self.regulator_optimizer = optax.adam(c.alpha_lr)
        self.regulator_state = self.regulator_optimizer.init(self.regulator_log_alpha)
        self.regulator_rng = np.random.default_rng(np.random.SeedSequence([self.seed or 0, 9212026]))
        self.regulator_key = jax.random.PRNGKey(int(self.seed or 0) + 9212026)
        self.regulator_next_update = 0
        self.regulator_count = 0
        self.regulator_entropy = float('nan')
        @jax.jit
        def draw(state, obs, key):
            keys = jax.random.split(key, c.samples)
            return jax.vmap(lambda k: OptiQPolicy.sample_action(state, obs, k), out_axes=1)(keys)
        self.regulator_draw = draw

    @property
    def regulator_noise_std(self):
        return float(self.regulator_cfg.noise_scale) * float(jnp.exp(self.regulator_log_alpha))

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        if not self.regulator_enabled or self.num_timesteps < learning_starts:
            return action, buffer_action
        std = self.regulator_noise_std
        noisy = noisy_action(buffer_action, self.regulator_rng.normal(size=buffer_action.shape), std)
        self.logger.record('exploration/noise_std', std)
        self.logger.record('exploration/clip_fraction', float(np.mean(np.abs(noisy) >= 1.)))
        # Store exactly the normalized action that is actually executed.
        return self.policy.unscale_action(noisy), noisy

    def train(self, batch_size, gradient_steps):
        if self.regulator_enabled and self._n_updates >= self.regulator_next_update:
            c = self.regulator_cfg
            started = time.monotonic()
            # Independent diagnostic replay RNG, restoring global state so this
            # extra draw does not displace the training batch's RNG sequence.
            saved = np.random.get_state()
            try:
                np.random.seed(int(self.regulator_rng.integers(0, 2**32 - 1)))
                data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            finally:
                np.random.set_state(saved)
            obs = data.observations.numpy()
            self.regulator_key, key = jax.random.split(self.regulator_key)
            actions = np.asarray(self.regulator_draw(self.policy.actor_state, jnp.asarray(obs), key))
            noise_std_before = self.regulator_noise_std
            actions = noisy_action(actions, self.regulator_rng.normal(size=actions.shape), noise_std_before)
            h, converged = entropy_proxy(actions, c.components, c.entropy_seed)
            target = float(c.target_entropy_per_dim) * actions.shape[-1]
            # Official log-alpha optimizer: dL/d(log alpha) = H_hat - H_target.
            updates, self.regulator_state = self.regulator_optimizer.update(
                jnp.asarray(h-target,dtype=jnp.float32), self.regulator_state, self.regulator_log_alpha)
            self.regulator_log_alpha = optax.apply_updates(self.regulator_log_alpha, updates)
            assert np.isfinite(float(self.regulator_log_alpha))
            self.regulator_entropy = h
            self.regulator_count += 1
            self.regulator_next_update += int(c.interval_updates)
            metrics = dict(entropy_proxy=h, target_entropy=target,
                           alpha=float(jnp.exp(self.regulator_log_alpha)), noise_std=self.regulator_noise_std,
                           noise_std_before=noise_std_before,
                           entropy_probe_clip_fraction=float(np.mean(np.abs(actions) >= 1.)),
                           updates=self.regulator_count, gmm_converged_fraction=converged,
                           estimation_seconds=time.monotonic()-started)
            for k,v in metrics.items(): self.logger.record('exploration/'+k, v)
            path = Path(self.cfg.output_root) / 'dacer_regulator.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            record=dict(metrics, env_steps=self.num_timesteps, learner_updates=self._n_updates)
            path.write_text(json.dumps(record,indent=2)+'\n')
            with path.with_name('dacer_regulator_history.jsonl').open('a') as history:
                history.write(json.dumps(record)+'\n')
        return super().train(batch_size, gradient_steps)
