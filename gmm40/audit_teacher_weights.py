"""Independent float64 audit of value weights and proposal density, CPU only."""
import hashlib
import importlib.util
import json
import math
import pickle

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scipy.special import logsumexp, softmax

from models.critic import VectorCritic
from models.utils import activation_fn
from optiq_dime.policy import SemiImplicitActor
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from .evaluation import atomic_json
from .optiq import OptiQ
from .target import ROOT, RESULTS, Target


def independent_logq(u, mu, ls, floor=.05):
    u, mu, ls = (np.asarray(x, dtype=np.float64) for x in (u, mu, ls))
    ls = np.maximum(ls, np.log(floor))
    component = (-.5*((u[:, :, None]-mu[:, None])/np.exp(ls[:, None]))**2
                 - ls[:, None] - .5*np.log(2*np.pi)).sum(-1)
    log_qu = logsumexp(component, axis=-1)-np.log(mu.shape[1])
    # sech(u)^2, computed independently using stable log(cosh).
    log_jac = (-2*(np.logaddexp(u, -u)-np.log(2))).sum(-1)
    return log_qu-log_jac, log_qu, log_jac


def summary(weights, q, high_density=None):
    result = dict(ess=float((1/(weights*weights).sum(-1)).mean()),
                  max_weight_mean=float(weights.max(-1).mean()),
                  weighted_Q=float((weights*q).sum(-1).mean()),
                  weighted_Q_gain=float(((weights*q).sum(-1)-q.mean(-1)).mean()))
    if high_density is not None:
        result['high_density_weighted_mass'] = float((weights*high_density).sum(-1).mean())
    return result


def audit_cloud(label, mu, ls, q_fn, temperature, target=None, seed=78129):
    proposal = ConditionalGaussianProposal(mu, ls, .05)
    key = jax.random.PRNGKey(seed)
    a, u, indices = proposal.sample(key, 4, 'exact')
    q = np.asarray(q_fn(a), dtype=np.float64)
    logq_jax = np.asarray(proposal.log_prob(u), dtype=np.float64)
    logqa, logqu, logjac = independent_logq(u, mu, ls)
    # Fixed-Q adapter uses physical-x density; moving uses normalized-action density.
    coordinate_constant = 2*np.log(40) if target is not None else 0.
    qdensity = logqa-coordinate_constant
    w_np = softmax(q/temperature-qdensity, axis=-1)
    w_actual = np.asarray(jax.nn.softmax(jnp.asarray(q)/temperature
                                        -(proposal.log_prob(u)-coordinate_constant), axis=-1))
    wp = softmax(q/temperature, axis=-1)
    wd = softmax(-qdensity, axis=-1)
    uniform = np.full(q.shape, 1/q.shape[1])
    qstd = np.std(q/temperature, axis=-1)
    dstd = np.std(-qdensity, axis=-1)
    high = None
    if target is not None:
        x = np.asarray(a)*40
        distance = (((x[:, :, None]-target.means)/target.std[None, None, :, None])**2).sum(-1)
        high = distance.min(-1) <= 9
    # Reconstruct the sampler using its independent component/noise keys and floor.
    component_key, noise_key = jax.random.split(key)
    expected_indices = jax.random.randint(component_key, q.shape, 0, 16)
    effective = np.maximum(np.asarray(ls), np.log(.05))
    selected_mu = np.take_along_axis(np.asarray(mu), np.asarray(indices)[..., None], axis=1)
    selected_ls = np.take_along_axis(effective, np.asarray(indices)[..., None], axis=1)
    eta = np.asarray(jax.random.normal(noise_key, u.shape, dtype=u.dtype))
    expected_u = selected_mu+np.exp(selected_ls)*eta
    counts = np.bincount(np.asarray(indices).ravel(), minlength=16)
    result = dict(label=label, states=int(q.shape[0]), candidates_per_state=64, temperature=temperature,
                  max_logq_abs_error=float(np.max(np.abs(logqa-logq_jax))),
                  max_weight_abs_error=float(np.max(np.abs(w_np-w_actual))),
                  max_weight_sum_error=float(np.max(np.abs(w_actual.sum(-1)-1))),
                  finite=bool(np.isfinite(w_actual).all() and np.isfinite(q).all() and np.isfinite(logqa).all()),
                  sampler_component_indices_equal=bool(np.array_equal(indices, expected_indices)),
                  sampler_max_u_abs_error=float(np.max(np.abs(expected_u-np.asarray(u)))),
                  component_draw_counts=counts.tolist(),
                  proposal_floor_fraction=float((np.asarray(ls)<np.log(.05)).mean()),
                  Q_logit_std_mean=float(qstd.mean()), density_logit_std_mean=float(dstd.mean()),
                  Q_range_mean=float(np.ptp(q, axis=-1).mean()),
                  Q_plus_constant_weight_error=float(np.max(np.abs(softmax((q+17)/temperature-qdensity, axis=-1)-w_np))),
                  uniform=summary(uniform, q, high), Q_only=summary(wp, q, high),
                  density_only=summary(wd, q, high), full=summary(w_np, q, high),
                  full_vs_Q_only_TV=float((.5*np.abs(w_np-wp).sum(-1)).mean()),
                  max_error_omitting_jacobian=float(np.max(np.abs(w_np-softmax(q/temperature-(logqu-coordinate_constant), axis=-1)))))
    if target is not None:
        ratio = np.exp(q-qdensity)
        per_cloud = ratio.mean(-1)
        result.update(physical_vs_action_weight_error=float(np.max(np.abs(w_np-softmax(q/temperature-logqa, axis=-1)))),
                      bounded_normalization_weight_error=float(np.max(np.abs(w_np-softmax((q-target.log_z)/temperature-qdensity, axis=-1)))),
                      estimated_box_mass=float(per_cloud.mean()),
                      box_mass_mc_standard_error=float(per_cloud.std(ddof=1)/np.sqrt(len(per_cloud))),
                      true_box_mass=float(np.exp(target.log_z)))
    arrays = dict(mu=np.asarray(mu), log_std=np.asarray(ls), teacher_u=np.asarray(u),
                  action=np.asarray(a), Q=q, log_q_action=logqa, weights=w_actual,
                  independent_weights=w_np, component_indices=np.asarray(indices))
    result['passed'] = (result['max_logq_abs_error'] < 1e-4
                        and result['max_weight_abs_error'] < 1e-4
                        and result['max_weight_sum_error'] < 2e-6
                        and result['sampler_max_u_abs_error'] < 2e-6
                        and result['sampler_component_indices_equal'] and result['finite'])
    return result, arrays


def main():
    assert jax.default_backend() == 'cpu'
    folder = RESULTS/'diagnostics/teacher_weights_audit_20260918'
    folder.mkdir(parents=True, exist_ok=False)
    target = Target()
    fixed = RESULTS/'optiq_n16_m64_seed0_100k/checkpoints/step_0100000.bin'
    nav = RESULTS/'navigation_optiq_T025_seed0_100k'
    nav_ckpt = nav/'checkpoints/update_0100000.bin'
    paths = [fixed, nav_ckpt, ROOT/'gmm40/optiq.py', ROOT/'optiq_dime/algorithm.py',
             ROOT/'optiq_dime/semi_implicit.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    agent = OptiQ(target)
    agent.restore(fixed)
    z = jax.random.normal(jax.random.PRNGKey(92154), (256*16, 2))
    mu, ls = agent.actor.apply({'params': agent.state.params}, jnp.zeros((256*16, 1)), z)
    first, arrays = audit_cloud('fixed_Q_100K', mu.reshape(256, 16, 2), ls.reshape(256, 16, 2),
                               lambda a: target.jax_log_prob(40*a), 1., target)
    # Compare Q to the actual upstream DiKL implementation as well as NumPy.
    import torch
    path = ROOT/'gmm40-baseline/DiKL/DiKL/energy/mog40.py'
    spec = importlib.util.spec_from_file_location('gmm40_original_for_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = module.GMM(2, 40, 40, log_var_scaling=1, seed=0, device='cpu')
    x = np.asarray(arrays['action']*40)
    with torch.no_grad():
        upstream_q = upstream.log_prob(torch.as_tensor(x)).numpy()
    first['Q_max_error_vs_upstream_DiKL'] = float(np.max(np.abs(arrays['Q']-upstream_q)))
    first['Q_max_error_vs_numpy_float64'] = float(np.max(np.abs(arrays['Q']-target.log_prob(x))))
    first['passed'] &= first['Q_max_error_vs_upstream_DiKL'] < 1e-4
    np.savez_compressed(folder/'fixed_Q_100K.npz', **arrays)

    cfg = OmegaConf.load(nav/'resolved_optiq.yaml')
    checkpoint = flax.serialization.msgpack_restore(nav_ckpt.read_bytes())
    replay = pickle.loads((nav/'checkpoints/update_0100000.replay.pkl').read_bytes())
    rng = np.random.default_rng(18347)
    indices = rng.integers(0, replay.buffer_size if replay.full else replay.pos, 256)
    obs = replay.observations[indices, 0]
    a_cfg, c_cfg, opt_cfg = cfg.alg.actor, cfg.alg.critic, cfg.alg.optimizer
    actor = SemiImplicitActor(2, tuple(a_cfg.hidden_dims), a_cfg.log_std_min, a_cfg.log_std_max,
                               a_cfg.initial_log_std, a_cfg.mean_output_init_scale,
                               a_cfg.log_std_output_init_scale)
    mu, ls = actor.apply({'params': checkpoint['actor']['params']},
                         jnp.repeat(jnp.asarray(obs), 16, axis=0), z)
    critic = VectorCritic(dropout_rate=c_cfg.dropout_rate, use_layer_norm=c_cfg.use_layer_norm,
                          use_batch_norm=opt_cfg.bn, bn_warmup=opt_cfg.bn_warmup,
                          batch_norm_momentum=opt_cfg.bn_momentum, batch_norm_mode=opt_cfg.bn_mode,
                          net_arch=c_cfg.hs, activation_fn=activation_fn[c_cfg.activation],
                          n_critics=2, n_atoms=1)
    def q_pair(actions):
        return critic.apply({'params': checkpoint['critic']['params'],
                             'batch_stats': checkpoint['critic']['batch_stats']},
                            jnp.repeat(jnp.asarray(obs), 64, axis=0), actions.reshape(-1, 2),
                            train=False).reshape(2, 256, 64)
    second, arrays = audit_cloud('moving_Q_100K_replay_states', mu.reshape(256,16,2), ls.reshape(256,16,2),
                                lambda a: q_pair(a).mean(0), float(a_cfg.temperature))
    pair = np.asarray(q_pair(jnp.asarray(arrays['action'])), dtype=np.float64)
    q1, q2 = pair
    density = arrays['log_q_action']
    w1, w2 = softmax(q1/.25-density, axis=-1), softmax(q2/.25-density, axis=-1)
    own_gain = .5*((w1*q1).sum(-1)-q1.mean(-1)+(w2*q2).sum(-1)-q2.mean(-1))
    cross_gain = .5*((w1*q2).sum(-1)-q2.mean(-1)+(w2*q1).sum(-1)-q1.mean(-1))
    second.update(live_twin_mean_verified=bool(np.allclose(pair.mean(0), arrays['Q'], atol=2e-5)),
                  critic_twin_action_gap_abs_mean=float(np.abs((q1-q1.mean(-1,keepdims=True))-(q2-q2.mean(-1,keepdims=True))).mean()),
                  twin_own_weighted_gain=float(own_gain.mean()), twin_cross_weighted_gain=float(cross_gain.mean()),
                  twin_argmax_agreement=float((q1.argmax(-1)==q2.argmax(-1)).mean()),
                  configured_density_beta=float(a_cfg.density_beta),
                  adaptive_density_beta=bool(a_cfg.adaptive_density_beta),
                  source_q_eval=str(a_cfg.source_q_eval),
                  state_source='256 fixed-seed samples from saved 100K replay buffer')
    np.savez_compressed(folder/'moving_Q_100K.npz', observations=obs, q1=q1, q2=q2, **arrays)

    # Exercise the teacher floor with deliberately narrower conditional sigmas.
    stressed_ls = ls.reshape(256,16,2)-4
    third, _ = audit_cloud('synthetic_floor_stress_not_training', mu.reshape(256,16,2), stressed_ls,
                            lambda a: jnp.zeros(a.shape[:-1]), 1.)
    cases = [first, second, third]
    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in paths)
    result = dict(passed=all(c['passed'] for c in cases), inputs_unchanged=unchanged,
                   input_sha256=hashes, optimizer_updates=0, backend=jax.default_backend(), cases=cases,
                   scope='Exact finite conditional proposal, not the intractable continuous-z policy marginal. '
                         'Finite self-normalized importance weights are not unbiased exact target samples. '
                         'Passing formulas does not certify the learned moving critic is correct.')
    atomic_json(folder/'results.json', result)
    for case in cases:
        print(json.dumps(case), flush=True)
    assert result['passed'] and unchanged, result


if __name__ == '__main__':
    main()
