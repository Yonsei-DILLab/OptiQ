"""Read-only native-density probes at the same start-state distribution."""
from __future__ import annotations

import numpy as np

from .envs import TaskBatch


def measure(agent, task, seed, count=512):
    env = TaskBatch(task, count=count, seed=seed)
    observations = env.current.copy()
    env.close()
    with agent.evaluation_rng(seed):
        if agent.method == "mfpo":
            actions, logp, _ = agent.agent.sample_actions_with_logp(observations)
            alpha = float(agent.agent.temp.apply_fn({"params": agent.agent.temp.params}))
            density_kind = "native learned flow log-density before final hard clipping"
            target = float(agent.agent.target_entropy)
        elif agent.method == "meow":
            import torch
            agent.policy.eval()
            with torch.no_grad():
                actions, logp = agent.policy.sample(count, observations, deterministic=False)
            actions, logp = actions.cpu().numpy(), logp.cpu().numpy()
            alpha = float(agent.policy.alpha)
            density_kind = "native transformed-flow log-density"
            target = None
        else:
            raise ValueError(agent.method)
        actions, logp = np.asarray(actions), np.asarray(logp)
        q = agent.q(observations, actions)
    if not all(np.isfinite(x).all() for x in (actions, logp, q)):
        raise FloatingPointError("nonfinite native entropy probe")
    entropy = float(-logp.mean())
    return dict(sample_count=count, seed=seed, density_kind=density_kind,
                target_entropy_total=target, entropy_estimate=entropy, alpha=alpha,
                alpha_entropy=alpha * entropy, q_mean=float(q.mean()), q_std=float(q.std()),
                action_mean=actions.mean(axis=0).tolist(), action_std=actions.std(axis=0).tolist(),
                action_covariance=np.cov(actions.T).tolist(),
                clipped_boundary_fraction=float(np.mean(np.abs(actions) >= .999)),
                start_xy_mean=observations[:, :2].mean(axis=0).tolist(),
                note="Action entropy is not goal or trajectory entropy; MFPO clipping density is approximate.")
