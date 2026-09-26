"""Original batched PointMaze rollout and five-trial robustness protocol."""
import numpy as np
from .batch import TaskBatch
from .metrics import removal_sr5_exact as removal_sr5
from .drac_paper import HORIZONS as PAPER_HORIZONS, GOAL_COUNTS as PAPER_GOALS
PAPER_TASKS = tuple("pm_" + name for name in PAPER_HORIZONS)

def obstacle_sr5(goals: np.ndarray) -> float:
    if len(goals) % 5:
        raise ValueError("five-trial robustness requires a multiple of five episodes")
    return float(np.mean(np.any(goals.reshape(-1, 5) >= 0, axis=1)))


def evaluate(agent, task, seed, episodes, mode, destination, obstacle=False):
    if task not in PAPER_TASKS:
        raise ValueError(task)
    if mode != "mu_only":
        raise ValueError("iBOLT rollouts require mu_only with fresh random z")
    if episodes <= 0 or episodes % 5:
        raise ValueError(episodes)
    horizon = PAPER_HORIZONS[task[3:]]
    dim = 4
    histories = np.full((episodes, horizon + 1, 2), np.nan, np.float32)
    returns = np.zeros(episodes, np.float32)
    goal_ids = np.full(episodes, -1, np.int8)
    lengths = np.zeros(episodes, np.int32)
    for offset in range(0, episodes, 128):
        stop = min(offset + 128, episodes)
        env = TaskBatch(task[3:], count=stop - offset, seed=seed + offset, obstacle=obstacle)
        active = np.ones(stop - offset, bool)
        history_chunk = histories[offset:stop]
        return_chunk = returns[offset:stop]
        goal_chunk = goal_ids[offset:stop]
        length_chunk = lengths[offset:stop]
        history_chunk[:, 0] = env.current[:, :2]
        try:
            with agent.evaluation_rng(seed + offset):
                for index in range(horizon):
                    if not active.any():
                        break
                    # Every active policy sees its own uninterrupted state path.
                    observations = env.current.copy()
                    actions = agent.act(observations, mode=mode)
                    next_obs, rewards, terminated, truncated, goals = env.step(actions, active)
                    history_chunk[active, index + 1] = next_obs[active, :2]
                    return_chunk[active] += rewards[active]
                    length_chunk[active] += 1
                    goal_chunk[active & terminated] = goals[active & terminated]
                    active &= ~(terminated | truncated)
        finally:
            env.close()
    if not np.isfinite(returns).all() or np.any(lengths == 0):
        raise FloatingPointError("invalid evaluation rollout")
    np.savez_compressed(destination, xy=histories, returns=returns,
                        lengths=lengths, goal_ids=goal_ids, mode=mode,
                        observation_dim=dim, obstacle=obstacle)
    goal_count = PAPER_GOALS[task[3:]]
    counts = np.bincount(goal_ids[goal_ids >= 0], minlength=goal_count)
    result = dict(episodes=episodes, success=float(np.mean(goal_ids >= 0)),
                goals=counts.tolist(), failure=int(np.sum(goal_ids < 0)),
                reachable_goals=int(np.count_nonzero(counts)),
                mean_return=float(returns.mean()), mean_length=float(lengths.mean()))
    if task in PAPER_TASKS:
        result["sr5_obstacle" if obstacle else "sr5_removal"] = (
            obstacle_sr5(goal_ids) if obstacle else removal_sr5(goal_ids, goal_count))
    return result
