"""CPU-only comparison with the production v5 actor update; no run mutation.

The two runners split PRNG keys differently. Only that split is aligned in the
adapter for this diagnostic, so the actual loss/Adam update can be compared on
identical draws. Checkpoints and training sources are read-only inputs.
"""
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax

from common.type_aliases import RLTrainState
from gmm40._v5.optiq_dime.algorithm import OptiQDIME
from gmm40._v5.optiq_dime.policy import SemiImplicitActor
from . import V5_SOURCE
from .evaluation import atomic_json
from .optiq import OptiQ
from .target import RESULTS, ROOT, Target


def max_difference(a, b):
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(jax.tree_util.tree_leaves(a),
                               jax.tree_util.tree_leaves(b)))


def main():
    assert jax.default_backend() == "cpu", "Diagnostic must not compete with GPU jobs"
    folder = RESULTS / "diagnostics/v5_adapter_parity_20260918"
    folder.mkdir(parents=True, exist_ok=False)
    target = Target()
    files = [ROOT / "gmm40/optiq.py", V5_SOURCE / "optiq_dime/algorithm.py",
             RESULTS / "optiq_n16_m64_seed0/checkpoints/step_0000000.bin",
             RESULTS / "optiq_n16_m64_seed0_100k/checkpoints/step_0100000.bin"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    def oracle(variables, observations, actions, **kwargs):
        q = target.jax_log_prob(40 * actions)
        return jnp.stack((q, q), axis=0)[..., None]

    critic = RLTrainState.create(apply_fn=oracle, params={}, target_params={},
                                batch_stats={}, target_batch_stats={}, tx=optax.adam(3e-4))
    results = []
    for checkpoint, epsilon, iterations in [(files[2], .1, 100), (files[3], .1, 100),
                                            (files[3], .03, 1000), (files[3], .01, 3000)]:
        agent = OptiQ(target, batch=32, epsilon=epsilon, sinkhorn_iterations=iterations)
        agent.restore(checkpoint)
        before_state = agent.state
        root_key = jax.random.PRNGKey(631824)
        new_key, latent_key, proposal_key, _ = jax.random.split(root_key, 4)
        z_key, eps_key = jax.random.split(latent_key)
        split_original = jax.random.split
        remaps = []

        def aligned_split(key, num=2):
            if num == 4 and not isinstance(key, jax.core.Tracer) and np.array_equal(key, root_key):
                remaps.append(True)
                return jnp.stack((new_key, z_key, eps_key, proposal_key))
            return split_original(key, num)

        # Eager bodies avoid comparisons being affected by separate JIT fusions.
        # Shared proposal, density, Sinkhorn, NLL, and Adam code are unmodified.
        production, loss, returned_key, info = OptiQDIME.update_actor.__wrapped__(
            before_state, critic, jnp.zeros((32, 1)), root_key, jnp.array([0.]),
            num_policy_samples=16, proposals_per_policy_sample=4,
            proposal_sampling_mode="exact", proposal_std=.05, proposal_clip=.5,
            include_anchor=False, density_correction=True, density_beta=1.,
            adaptive_density_beta=False, minimum_source_ess=16., density_beta_grid_size=257,
            temperature=1., sinkhorn_epsilon=epsilon, sinkhorn_iterations=iterations,
            source_q_eval="mean", transport_target_mode="argmax", semi_implicit=True,
            normalize_ot_cost=False, distillation_loss="conditional_ot_nll",
            teacher_distribution="conditional_mixture", soft_proximal_ess_fraction=0.,
            entropy_diagnostics=False, ot_student_action="mean")
        with patch("jax.random.split", aligned_split):
            (adapter, adapter_key), adapter_info = agent._update((before_state, root_key), None)
        assert len(remaps) == 1
        values = dict(checkpoint=str(checkpoint), training_updates=agent.updates,
                      epsilon=epsilon, sinkhorn_iterations=iterations, batch=32, n=16, m=64,
                      production_loss=float(loss), adapter_loss=float(adapter_info["loss"]),
                      loss_abs_diff=float(abs(loss - adapter_info["loss"])),
                      params_max_abs_diff=max_difference(production.params, adapter.params),
                      optimizer_max_abs_diff=max_difference(production.opt_state, adapter.opt_state),
                      teacher_ess_abs_diff=float(abs(info["source_ess_absolute"] - adapter_info["teacher_ess"])),
                      next_key_equal=bool(np.array_equal(returned_key, adapter_key)),
                      optimizer_steps_equal=int(production.step) == int(adapter.step))
        values["passed"] = (values["loss_abs_diff"] < 2e-5
                            and values["params_max_abs_diff"] < 2e-5
                            and values["optimizer_max_abs_diff"] < 2e-5
                            and values["teacher_ess_abs_diff"] < 2e-5
                            and values["next_key_equal"] and values["optimizer_steps_equal"])
        results.append(values)
        print(json.dumps(values), flush=True)

    # Initialization sensitivity is measured without performing training.
    init_key = jax.random.split(jax.random.PRNGKey(0))[1]
    z = jax.random.normal(jax.random.PRNGKey(54687), (4096, 2))
    initialization = []
    for scale in [1e-4, 1e-3, 1e-2, 1e-1]:
        model = SemiImplicitActor(2, (256, 256), -5., 1., math.log(.5),
                                  mean_output_init_scale=scale)
        params = model.init(init_key, jnp.zeros((1, 1)), jnp.zeros((1, 2)))["params"]
        mu, ls = model.apply({"params": params}, jnp.zeros((4096, 1)), z)
        initialization.append(dict(mean_head_variance_scale=scale,
                                   initial_mean_position_std_xy=np.std(40*np.tanh(mu), axis=0).tolist(),
                                   initial_sigma_mean=float(jnp.exp(ls).mean())))

    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in files)
    payload = dict(passed=all(x["passed"] for x in results), backend=jax.default_backend(),
                   inputs_unchanged=unchanged, input_sha256=hashes, cases=results,
                   initialization_without_training=initialization,
                   scope="One temporary CPU Adam step per runner/case, same starting parameters and draws. "
                         "No saved model or live run changed. Eager-body check; not GPU/100K trajectory parity. "
                         "Production and adapter use equivalent densities differing by the constant 2*log(40).",
                   tolerances=dict(loss=2e-5, params=2e-5, optimizer_state=2e-5, teacher_ess=2e-5))
    atomic_json(folder / "results.json", payload)
    assert unchanged and payload["passed"], payload


if __name__ == "__main__":
    main()
