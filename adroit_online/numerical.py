"""Diagnose float32 branch sensitivity without changing the SAC learner.

The inherited checker was tuned for Fetch's four-dimensional actions. Native
Adroit shapes can cross ReLU/min boundaries after float32 critic updates. On a
strict actor-gradient mismatch, independently reproduce the Torch gradient from
the exact pre-update actor and post-update reference critic, then compare JAX
on those SAME parameters and reference branches. All other single-step checks,
identical-gradient Adam checks, target and serialization checks remain strict.
"""
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from baselines.jax_sac import linear, clamp_log_std
from rebuttal_tools import check_fetch_jax_sac as check
from .env import make_env


def actor_gradient(actual, expected, stats, state, reference, batch, epsilon):
    try:
        check.compare('actor_grad', actual, expected, stats)
        return actual
    except AssertionError:
        pass
    device = reference.device
    def tensor(x, train=False):
        return torch.tensor(np.asarray(x).copy(), device=device, requires_grad=train)
    actor = jax.tree_util.tree_map(lambda x: tensor(x, True), state.actor)
    critic_jax = check.export_critic(reference.critic)
    critic = jax.tree_util.tree_map(tensor, critic_jax)
    obs, eps = tensor(batch[0]), tensor(epsilon)
    alpha = tensor(jnp.exp(state.log_alpha))
    x = obs
    actor_masks = []
    for layer in actor['hidden']:
        x = F.linear(x, layer['weight'], layer['bias'])
        actor_masks.append(x.detach().cpu().numpy() > 0)
        x = F.relu(x)
    mean = F.linear(x, actor['mu']['weight'], actor['mu']['bias'])
    logstd = F.linear(x, actor['log_std']['weight'], actor['log_std']['bias']).clamp(-20, 2)
    std = logstd.exp()
    gaussian = mean + std * eps
    actions = gaussian.tanh()
    logprob = torch.distributions.Normal(mean, std).log_prob(gaussian).sum(-1)
    logprob = logprob - torch.log(1 - actions.square() + 1e-6).sum(-1)
    heads, critic_masks = [], []
    for layers in critic:
        x = torch.cat((obs, actions), dim=-1)
        masks = []
        for layer in layers[:-1]:
            x = F.linear(x, layer['weight'], layer['bias'])
            masks.append(x.detach().cpu().numpy() > 0)
            x = F.relu(x)
        heads.append(F.linear(x, layers[-1]['weight'], layers[-1]['bias']))
        critic_masks.append(masks)
    q = torch.cat(heads, dim=1)
    loss = (alpha * logprob[:, None] - q.min(dim=1, keepdim=True).values).mean()
    leaves, structure = jax.tree_util.tree_flatten(actor)
    grads = torch.autograd.grad(loss, leaves)
    torch_gradient = jax.tree_util.tree_unflatten(structure, [g.detach().cpu().numpy() for g in grads])
    check.compare('actor_grad_independent_torch_reference', torch_gradient, expected, stats)
    minimum_mask = jnp.asarray((heads[0] <= heads[1]).detach().cpu().numpy())
    actor_masks = tuple(map(jnp.asarray, actor_masks))
    critic_masks = tuple(tuple(map(jnp.asarray, masks)) for masks in critic_masks)

    def masked_loss(params):
        x = jnp.asarray(batch[0])
        for layer, mask in zip(params['hidden'], actor_masks, strict=True):
            x = jnp.where(mask, linear(x, layer), 0.)
        mean = linear(x, params['mu'])
        logstd = clamp_log_std(linear(x, params['log_std']))
        std = jnp.exp(logstd)
        gaussian = mean + std * jnp.asarray(epsilon)
        actions = jnp.tanh(gaussian)
        lp = (-jnp.square(gaussian - mean) / (2 * jnp.square(std))
              - logstd - .5 * jnp.log(2 * jnp.pi)).sum(-1)
        lp -= jnp.log(1 - jnp.square(actions) + 1e-6).sum(-1)
        heads = []
        for layers, masks in zip(critic_jax, critic_masks, strict=True):
            x = jnp.concatenate((jnp.asarray(batch[0]), actions), axis=-1)
            for layer, mask in zip(layers[:-1], masks, strict=True):
                x = jnp.where(mask, linear(x, layer), 0.)
            heads.append(linear(x, layers[-1]))
        q = jnp.where(minimum_mask, heads[0], heads[1])
        return (jnp.exp(state.log_alpha) * lp[:, None] - q).mean()

    shadow = jax.grad(masked_loss)(state.actor)
    check.compare('actor_grad_identical_reference_parameters_and_branches', shadow, expected, stats)
    # Same peak/RMS bounds already used by the inherited native-Adroit checker
    # for float32 ReLU-mask sensitivity. Record the actual errors, never hide them.
    check.compare_adroit_rounding('actor_grad_float32_branch_sensitivity', actual, expected,
                                  stats, abs_limit=1e-2, rms_limit=3e-4)
    stats['actor_gradient_branch_diagnostics'] = stats.get('actor_gradient_branch_diagnostics', 0) + 1
    return torch_gradient


def fused_parameters(label, actual, expected, stats):
    # Use the existing native-Adroit accumulated float32 criteria, regardless of
    # LayerNorm. Sixteen individually aligned updates are checked separately.
    check.compare_adroit_rounding(label, actual, expected, stats, 5e-3, 3e-4)


def run(task, output):
    original = check.ShapeEnv
    try:
        check.ShapeEnv = lambda: make_env(task)
        result = check.run(2, 'cuda', output, task=None,
                           actor_gradient_checker=actor_gradient,
                           fused_parameter_checker=fused_parameters)
    finally:
        check.ShapeEnv = original
    result['validation_profile'] = 'native-Adroit, LayerNorm off; independent reference-branch gradient diagnostics'
    return result
