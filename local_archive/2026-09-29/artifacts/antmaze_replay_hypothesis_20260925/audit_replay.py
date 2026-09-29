"""Read-only CPU audit of an unwritten-over AntMaze replay checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def route(task, xy):
    if task == 'v3':
        left, right = (xy[:, 0] < -8).any(), (xy[:, 0] > 8).any()
        return 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'
    ix = np.flatnonzero((xy[:-1, 0] > -4) & (xy[1:, 0] <= -4))
    if not len(ix):
        return 'uncommitted'
    i = ix[0]
    u = (-4 - xy[i, 0]) / (xy[i + 1, 0] - xy[i, 0])
    y = xy[i, 1] + u * (xy[i + 1, 1] - xy[i, 1])
    return 'upper' if y > 2 else 'lower' if y < -2 else 'uncommitted'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expect-sha', required=True)
    a = p.parse_args()
    with a.checkpoint.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    assert digest == a.expect_sha
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    c, meta = state['config'], state['replay_metadata']
    task, n, envs = c['task'], state['step'], c['num_envs']
    assert task in ('v3', 'v4') and envs == 256
    assert not meta['if_full']
    assert meta['cur_capacity'] == meta['total_samples'] == meta['next_p'] == n
    assert n < c['native']['alg']['buffer_size'] == 1000000
    replay = {k: v.numpy() for k, v in state['replay'].items()}
    obs, nxt, done = replay['buf_obs'], replay['buf_next_obs'], replay['buf_done'].reshape(-1)
    assert obs.shape == nxt.shape == (n, 29)
    assert np.isfinite(obs).all() and np.isfinite(nxt).all()
    assert n % envs == 0
    xy = obs[:, :2]
    masks = ({'left': xy[:, 0] < -8, 'right': xy[:, 0] > 8} if task == 'v3' else
             {'upper': (xy[:, 0] < -4) & (xy[:, 1] > 2),
              'lower': (xy[:, 0] < -4) & (xy[:, 1] < -2)})
    ticks = sorted(set([0, n] + [((x + 255) // 256) * 256 for x in range(25000, n, 25000)]))
    occupancy = []
    for t in ticks[1:]:
        start = max(0, t - 50176)
        occupancy.append(dict(step=t, retained=t, overwritten=0, recent_start=start,
            cumulative_counts={k: int(m[:t].sum()) for k, m in masks.items()},
            recent_counts={k: int(m[start:t].sum()) for k, m in masks.items()}))

    o = obs.reshape(-1, envs, 29)
    no = nxt.reshape(-1, envs, 29)
    d = done.reshape(-1, envs)
    final_obs = np.asarray(state['observations'])
    reset = np.any(o[1:] != no[:-1], axis=-1)
    final_reset = np.any(final_obs != no[-1], axis=-1)
    episodes, partial = [], []
    for env in range(envs):
        stops = (np.flatnonzero(reset[:, env]) + 1).tolist()
        if final_reset[env]:
            stops.append(len(o))
        start = 0
        for end in stops:
            terminal = bool(d[end - 1, env])
            assert terminal or end - start == 700, (task, env, start, end)
            assert not d[start:end - 1, env].any()
            track = np.concatenate([o[start:start + 1, env, :2], no[start:end, env, :2]])
            episodes.append(dict(env=env, begin=start * envs + env + 1,
                end=(end - 1) * envs + env + 1, length=end - start,
                route=route(task, track), success=terminal))
            start = end
        if start < len(o):
            track = np.concatenate([o[start:start + 1, env, :2], no[start:, env, :2]])
            partial.append(dict(env=env, begin=start * envs + env + 1,
                                route=route(task, track), length=len(o) - start))
    assert sum(e['length'] for e in episodes + partial) == n
    assert sum(e['success'] for e in episodes) == int(done.sum())
    regions = {name: dict(count=int(mask.sum()), fraction=float(mask.mean()))
               for name, mask in masks.items()}
    result = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=digest,
        bytes=a.checkpoint.stat().st_size, source_commit=c['source_commit'],
        task=task, step=n, updates=state['updates'], replay_metadata=meta,
        max_capacity=1000000, overwritten_transitions=0, chronological_order_verified=True,
        final_regions=regions, occupancy=occupancy,
        completed_episodes=len(episodes), completed_route_counts=dict(Counter(e['route'] for e in episodes)),
        successful_route_counts=dict(Counter(e['route'] for e in episodes if e['success'])),
        incomplete_route_counts=dict(Counter(e['route'] for e in partial)),
        episodes=episodes, incomplete_episodes=partial,
        region_definition='v3: x<-8 / x>8; v4: x<-4 and y>2 / y<-2. Region counts are transitions, not episodes.',
        episode_definition='Per-environment reset detected by exact 29D next-observation continuity; termination or native700 horizon checked.',
        all_data_retained=True, no_training_or_rollouts=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
