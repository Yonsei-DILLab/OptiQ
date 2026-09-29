"""Read-only branch progress; no rollout, timeout, reward, or training changes."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    report = json.loads((ROOT / 'report/results.json').read_text())
    rows = []
    for row in report['history']:
        if row['mode'] != 'policy' or row['step'] < 200000:
            continue
        raw = Path(row['raw_path'])
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row['raw_sha256']
        data = np.load(raw)
        groups = {}
        for xy, n, goal_id in zip(data['xy'], data['lengths'], data['goals']):
            n = int(n)
            path = xy[:n + 1]
            left, right = (path[:, 0] < -8).any(), (path[:, 0] > 8).any()
            side = 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'
            targets = np.array([[-12., 12.], [12., -12.]])
            if side in ('left', 'right'):
                targets = targets[:1] if side == 'left' else targets[1:]
            distances = np.linalg.norm(path[:, None, :] - targets[None, :, :], axis=-1).min(axis=1)
            groups.setdefault(side, []).append([
                n, int(goal_id) > 0, distances.min(), distances[-1],
                distances[max(0, n - 100)] - distances[-1]])
        summaries = []
        for side, values in groups.items():
            a = np.asarray(values, np.float64)
            summaries.append(dict(
                side=side, episodes=len(a), successes=int(a[:, 1].sum()),
                at_original_700_step_limit=int((a[:, 0] == 700).sum()),
                episode_length_mean=float(a[:, 0].mean()),
                closest_goal_distance_mean=float(a[:, 2].mean()),
                final_goal_distance_mean=float(a[:, 3].mean()),
                last100_distance_gain_mean=float(a[:, 4].mean()),
                positive_last100_distance_gain_count=int((a[:, 4] > 0).sum())))
        assert sum(g['episodes'] for g in summaries) == row['episodes']
        rows.append(dict(condition=row['condition'], step=row['step'],
                         episodes=row['episodes'], groups=summaries,
                         raw_path=str(raw), raw_sha256=row['raw_sha256']))
    result = dict(
        time_utc=datetime.now(timezone.utc).isoformat(),
        reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence='Existing direct-policy trajectories, original fixed full state, including failures',
        limitation='Retrospective path groups from one training seed. Reaching the time limit does not establish that extra time would produce success. No horizon override or extrapolated success.',
        rows=rows)
    output = ROOT / 'report/route-progress-diagnostic.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(path=str(output), rows=len(rows))))


if __name__ == '__main__':
    main()
