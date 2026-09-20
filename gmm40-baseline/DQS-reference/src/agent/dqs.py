import functools

import jax
import jax.numpy as jnp
import numpy as onp
import optax

from src.components.lambda_weighter import BasicLambdaWeighter
from src.components.score_estimator import estimate_grad_Rt, wrap_for_richardsons
from src.components.sde_integration import integrate_sde, integrate_pfode
from src.components.noise_schedule import noise_h
from src.components.prior import Prior
from src.components.mlp import ScoreMLP
from src.agent.critic import ReturnFunction
from src.utils.grad_utils import l2_norm, clip_grads


def anneal_temperature(start, end, step, num_steps):
    if step >= num_steps:
        return jnp.exp(end)
    else:
        log_temp = start + (end - start) * (step / num_steps)
        return jnp.exp(log_temp)


@functools.partial(jax.jit, static_argnames=(
    'module', 'optimizer'))
def update_scores(module, params, times, actions, states, estimated_scores, 
           lambda_weights, optimizer, optimizer_state, temps):
    def loss_fn(params):
        predicted_scores = module.apply(params, times, actions, states, temps)
        error_norms = jnp.mean(((predicted_scores - estimated_scores) ** 2), axis=-1)
        dqs_loss = jnp.mean(lambda_weights * error_norms)
        return dqs_loss

    loss, grad = jax.value_and_grad(loss_fn)(params)
    # grad = clip_grads(grad, 0.5)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, optimizer_state, l2_norm(grad)


@functools.partial(jax.jit, static_argnames=(
    'module', 'optimizer', 'input_shape'))
def init_fn(module, optimizer, input_shape, rng):
    dummy_input = jnp.ones((1, *input_shape))
    params = module.init(rng, dummy_input)
    opt_state = optimizer.init(params)
    return opt_state, params


class DQS:
    def __init__(
        self,
        sample_dim,
        state_dim,
        energy_fn_config,
        num_estimator_mc_samples,
        num_integration_steps,
        num_samples_to_sample_from_buffer,
        buffer_update_period,
        use_richardsons,
        sigma_min=0.00001,
        sigma_max=1.0,
        init_temperature=0.05,
        final_temperature=0.05,
        temperature_steps=int(1e6),
        diffusion_scale=1.0,
        warm_start_steps=1,
        seed=0,
    ):
        self.key = jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.net = ScoreMLP(256, 2, 128, sample_dim)

        self.net_params = self.net.init(self.key, jnp.zeros((1,)), jnp.zeros((1,sample_dim)),
                                        jnp.zeros((1,state_dim)), jnp.zeros((1,)))
        self.net_optimizer = optax.adam(3e-4)
        # schedule = optax.cosine_decay_schedule(
        #     init_value=1e-3,
        #     decay_steps=int(1e6),
        #     alpha=1e-4,
        # )
        # self.net_optimizer = optax.chain(
        #     # optax.clip(1.0),
        #     optax.adam(learning_rate=schedule),
        # )
        self.net_optimizer_state = self.net_optimizer.init(self.net_params)

        self.energy_function = ReturnFunction(**energy_fn_config)

        # Select gradient estimator (optionally with Richardson extrapolation)
        self.estimate_grad = estimate_grad_Rt
        if use_richardsons:
            self.estimate_grad = wrap_for_richardsons(self.estimate_grad)

        self.sigma_min = sigma_min
        self.sigma_diff = sigma_max / sigma_min
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.temperature_steps = temperature_steps

        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.buffer_update_period = buffer_update_period

        self.lambda_weighter = BasicLambdaWeighter(self.sigma_min, self.sigma_diff, epsilon=1e-3)
        self.prior = Prior(dim=sample_dim, 
                            scale=noise_h(1, self.sigma_min, self.sigma_diff) ** 0.5)
        self.diffusion_scale = diffusion_scale
        self.warm_start_steps = warm_start_steps

        self.step = 0

    def generate_samples(
        self,
        num_samples,
        cond=None,
        diffusion_scale=None,
        temperature=None,
        use_pfode=False
    ):
        diffusion_scale = diffusion_scale or self.diffusion_scale
        samples = self.prior.sample(num_samples)

        if not use_pfode:
            final_samples, self.key = integrate_sde(
                self.key,
                self.net,
                self.net_params,
                samples,
                cond,
                self.num_integration_steps,
                self.sigma_min,
                self.sigma_diff,
                diffusion_scale=diffusion_scale,
                temperature=temperature,
            )
        else:
            final_samples, self.key = integrate_pfode(
                self.key,
                self.net,
                self.net_params,
                samples,
                cond,
                self.num_integration_steps,
                self.sigma_min,
                self.sigma_diff,
                diffusion_scale=diffusion_scale,
                temperature=temperature,
            )
        return final_samples

    def sample_actions(self, states, diffusion_scale=1.0):
        if states.ndim == 1:
            states = jnp.expand_dims(states, 0)
        num_samples = states.shape[0]

        temperature = anneal_temperature(onp.log(self.init_temperature),
                                  onp.log(self.final_temperature), 
                                  self.step, 
                                  self.temperature_steps)
        temperature_array = jnp.ones(num_samples) * temperature
        actions = self.generate_samples(num_samples, cond=states,
                                        diffusion_scale=diffusion_scale,
                                        temperature=temperature_array)
        if actions.shape[0] == 1:
            actions = actions[0]
        return onp.tanh(actions), temperature

    def update(self, states, actions, rewards, masks, next_states):

        self.step += 1

        next_actions, temperature = self.sample_actions(next_states)
        train_metrics = self.energy_function.update(states, actions, rewards, 
                                                    masks, next_states, next_actions)
        
        self.key, key1, key2, key3 = jax.random.split(self.key, 4)

        times = jax.random.uniform(
            key1, (self.num_samples_to_sample_from_buffer,),
        )
        temperature_array = jnp.ones(self.num_samples_to_sample_from_buffer) * temperature

        noised_actions = actions + (
            jax.random.normal(key2, actions.shape) \
                * jnp.expand_dims(jnp.sqrt(noise_h(times, self.sigma_min, self.sigma_diff)), -1)
        )

        estimated_scores = self.estimate_grad(
            key3,
            times,
            noised_actions,
            states,
            self.energy_function,
            self.energy_function.get_params(),
            num_mc_samples=self.num_estimator_mc_samples,
            sigma_min=self.sigma_min,
            sigma_diff=self.sigma_diff,
            temperature=temperature_array,
        )

        lambda_weights = self.lambda_weighter(times)

        dqs_loss, self.net_params, self.net_optimizer_state, grad_norm = update_scores(
            self.net, 
            self.net_params, 
            times, 
            noised_actions, 
            states,
            estimated_scores, 
            lambda_weights, 
            self.net_optimizer,
            self.net_optimizer_state,
            temperature_array,
        )

        train_metrics.update({
            "dqs_loss": dqs_loss,
            "dqs_grad_norm": grad_norm,
        })
    
        return train_metrics
