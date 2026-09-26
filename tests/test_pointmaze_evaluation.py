from contextlib import contextmanager
from itertools import combinations
import numpy as np
import pytest
import pointmaze.evaluation as evaluation
from pointmaze.metrics import removal_sr5_exact


def test_removal_metric_exhaustive():
    for count in (4, 8):
        for distinct in range(min(count, 5) + 1):
            goals = list(range(distinct)) + [-1] * (5-distinct)
            expected = np.mean([any(g >= 0 and g not in removed for g in goals)
                for removed in combinations(range(count), count//2)])
            assert removal_sr5_exact(goals, count) == pytest.approx(expected)
    assert removal_sr5_exact([0,1,0,1,-1], 4) == pytest.approx(5/6)
    assert removal_sr5_exact([-1]*5 + [0]*5, 4) == .25
    assert evaluation.obstacle_sr5(np.array([-1]*5 + [0]*5)) == .5
    for bad in ([0,1], [-2,0,1,2,3], [0,1,2,3,4]):
        with pytest.raises(ValueError):
            removal_sr5_exact(bad,4)


def test_batch_boundary_and_completed_paths(tmp_path, monkeypatch):
    constructed, action_shapes, seeds = [], [], []

    class Batch:
        def __init__(self, maze, count, seed, obstacle=False):
            constructed.append((count, seed, obstacle))
            self.count = count
            self.current = np.zeros((count,4), np.float32)
            self.target_lengths = 1 + np.arange(count) % 3
            self.t = 0
        def step(self, actions, active):
            self.t += 1
            self.current[active,0] += 1
            done = active & (self.target_lengths == self.t)
            goals = np.where(done, np.arange(self.count)%4, -1).astype(np.int8)
            return self.current.copy(), done.astype(np.float32)*100, done, np.zeros(self.count,bool), goals
        def close(self):
            pass

    class Agent:
        method = 'ibolt'
        @contextmanager
        def evaluation_rng(self, seed):
            seeds.append(seed)
            yield
        def act(self, obs, mode):
            assert mode == 'mu_only'
            action_shapes.append(len(obs))
            return np.zeros((len(obs),2),np.float32)

    monkeypatch.setattr(evaluation, 'TaskBatch', Batch)
    output = tmp_path/'rollout.npz'
    report = evaluation.evaluate(Agent(), 'pm_simple', 217000, 130, 'mu_only', output)
    assert constructed == [(128,217000,False),(2,217128,False)]
    assert seeds == [217000,217128]
    assert action_shapes == [128,128,128,2,2]
    assert report['success'] == 1 and report['mean_return'] == 100
    assert sum(report['goals']) == 130
    with np.load(output) as raw:
        assert set(raw.files) == {'xy','returns','lengths','goal_ids','mode','observation_dim','obstacle'}
        for path, length in zip(raw['xy'], raw['lengths']):
            assert np.isfinite(path[:length+1]).all()
            assert np.isnan(path[length+1:]).all()
