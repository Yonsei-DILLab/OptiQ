"""Paired evaluation seeds, isolated learner RNG, success and return records."""
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
import numpy as np
from .env import make_env


@contextmanager
def isolated_rng(seed):
    py, np_state = random.getstate(), np.random.get_state()
    torch = sys.modules.get("torch")
    states = None
    if torch is not None:
        states = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        torch.manual_seed(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py); np.random.set_state(np_state)
        if states is not None:
            torch.set_rng_state(states[0]); torch.cuda.set_rng_state_all(states[1])


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class Evaluator:
    def __init__(self, folder, seed, episodes=10):
        self.folder = Path(folder); self.seed = seed
        self.envs = [make_env() for _ in range(episodes)]
        self.histories = {}

    def evaluate(self, agent, step, modes):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 4404]))
        seeds = rng.integers(0, 2**30, len(self.envs))
        policy_seed = int(rng.integers(0, 2**30))
        metrics = {}
        for mode in modes:
            returns = np.zeros(len(seeds)); lengths = np.zeros(len(seeds), int)
            successes = np.zeros(len(seeds), bool); finished = np.zeros(len(seeds), bool)
            obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(self.envs, seeds)])
            distance = np.linalg.norm(obs[:, -4:-2] - obs[:, -2:], axis=1)
            min_distance = distance.copy(); paths = [obs[0, -4:].copy()]
            with isolated_rng(policy_seed), agent.evaluation(mode, policy_seed):
                while not finished.all():
                    actions = np.asarray(agent.act(obs), np.float32)
                    if actions.shape != (len(seeds), 8) or not np.isfinite(actions).all():
                        raise FloatingPointError(f"Invalid evaluation actions: {actions.shape}")
                    for i, e in enumerate(self.envs):
                        if finished[i]: continue
                        obs[i], r, terminated, truncated, info = e.step(np.clip(actions[i], -1, 1))
                        returns[i] += r; lengths[i] += 1
                        successes[i] |= bool(info.get("success", False))
                        distance[i] = np.linalg.norm(obs[i, -4:-2] - obs[i, -2:])
                        min_distance[i] = min(min_distance[i], distance[i])
                        finished[i] = terminated or truncated
                        if i == 0: paths.append(obs[0, -4:].copy())
            values = dict(timesteps=step, results=returns, ep_lengths=lengths,
                          successes=successes, final_distances=distance, min_distances=min_distance,
                          env_seeds=seeds, policy_seed=policy_seed)
            if not all(np.isfinite(v).all() for v in values.values()):
                raise FloatingPointError("Nonfinite evaluation")
            h = self.histories.setdefault(mode, {k: [] for k in values})
            for k, v in values.items(): h[k].append(v)
            tmp = self.folder / f"evaluations_{mode}.tmp.npz"
            np.savez_compressed(tmp, **h)
            tmp.replace(self.folder / f"evaluations_{mode}.npz")
            if step in (0, 500000, 1000000):
                np.savez_compressed(self.folder / f"trajectory_{mode}_{step}.npz", path=paths, env_seed=seeds[0])
            metrics.update({f"eval/{mode}/{k}": float(np.mean(v)) for k, v in dict(
                mean_reward=returns, success_rate=successes, mean_ep_length=lengths,
                final_distance=distance, min_distance=min_distance).items()})
        return metrics

    def close(self):
        for e in self.envs: e.close()
