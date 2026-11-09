"""Post-hoc evaluation corrections; never change frozen learning or raw files."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def removal_sr5_exact(goals, goal_count):
    goals = np.asarray(goals)
    if goals.ndim != 1 or len(goals) == 0 or len(goals) % 5 or goal_count < 2:
        raise ValueError("SR5 requires nonempty five-episode groups and at least two goals")
    if np.any((goals < -1) | (goals >= goal_count)):
        raise ValueError("goal ID outside the declared map")
    removed = goal_count // 2
    denominator = math.comb(goal_count, removed)
    scores = []
    for group in goals.reshape(-1, 5):
        reached = len(set(int(goal) for goal in group if goal >= 0))
        # Failure means every reached goal is in the uniformly removed subset.
        failure_sets = (math.comb(goal_count - reached, removed - reached)
                        if reached <= removed else 0)
        scores.append(1. - failure_sets / denominator)
    return float(np.mean(scores))


def removal_from_raw(folder: Path, record, mode="policy"):
    with np.load(folder / f"{record['step']:09d}_{mode}.npz") as raw:
        ids = raw["goal_ids"]
        goals = record[mode]["goals"]
        if (ids.shape != (record[mode]["episodes"],) or
                np.bincount(ids[ids >= 0], minlength=len(goals)).tolist() != goals):
            raise ValueError("raw goal counts differ from saved summary")
        return removal_sr5_exact(ids, len(goals))
