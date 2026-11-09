"""Read-only reviewer metrics, checkpoint-only full-policy evaluation and monitoring."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from .guards import ROOT, assert_output_boundary, assert_work_branch

METHODS = ('ibolt', 'sac', 'sql', 'td3', 'dipo', 'mfpo', 'meow')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stats(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def goal_metrics(result):
    counts = np.asarray(result['goals'], dtype=np.int64)
    assert len(counts) in (4, 8) and np.all(counts >= 0)
    assert counts.sum() + result['failure'] == result['episodes']
    assert int(np.count_nonzero(counts)) == result['reachable_goals']
    assert abs(float(counts.sum() / result['episodes']) - result['success']) < 1e-7
    p = counts / counts.sum() if counts.sum() else np.zeros(len(counts))
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0]))) if counts.sum() else None
    return dict(result, goal_proportions_of_successes=p.tolist(),
                normalized_goal_entropy=entropy / math.log(len(counts)) if entropy is not None else None,
                effective_goals=math.exp(entropy) if entropy is not None else None,
                minimum_goal_count=int(counts.min()),
                coverage_definition='number of goals reached at least once within this training seed')


def discovery_metrics(rewards, next_states, goals, num_envs, warmup):
    """Inputs are the complete chronological environment history, not replay samples."""
    rewards = np.asarray(rewards).reshape(-1)
    hits = np.flatnonzero(rewards > 0)
    assert np.isfinite(rewards).all() and np.all(np.isin(rewards, [0., 100.]))
    total, count = len(rewards), len(goals)
    first = lambda idx: int(math.ceil((int(idx) + 1) / num_envs) * num_envs)
    first_by_goal = [None] * count
    counts = np.zeros(count, dtype=np.int64)
    if len(hits):
        distances = np.max(np.abs(next_states[hits, None, :2] - goals[None]), axis=2)
        labels = distances.argmin(axis=1)
        assert np.all(distances[np.arange(len(labels)), labels] <= .50001)
        counts = np.bincount(labels, minlength=count)
        for goal in range(count):
            indices = np.flatnonzero(labels == goal)
            if len(indices):
                first_by_goal[goal] = first(hits[indices[0]])
    grid = sorted({0, warmup, total} | {
        min(total, int(math.ceil(t / num_envs) * num_envs)) for t in range(10000, total, 10000)})
    cumulative_rewards = np.searchsorted(hits, grid, side='left')
    cumulative_goals = [sum(t is not None and t <= step for t in first_by_goal) for step in grid]
    frequencies = []
    for i in range(1, len(grid)):
        frequencies.append(dict(start=grid[i - 1], end=grid[i],
            reward_events=int(cumulative_rewards[i] - cumulative_rewards[i - 1]),
            reward_events_per_100000_transitions=float(
                100000 * (cumulative_rewards[i] - cumulative_rewards[i - 1]) / (grid[i] - grid[i - 1]))))
    return dict(total_transitions=total, reward_discovered=bool(len(hits)),
                first_reward_at_collection_end=first(hits[0]) if len(hits) else None,
                first_reward_resolution_transitions=num_envs,
                first_goal_reward_at_collection_end=first_by_goal,
                all_goals_discovered=all(t is not None for t in first_by_goal),
                all_goals_discovery_at_collection_end=max(first_by_goal) if all(t is not None for t in first_by_goal) else None,
                warmup_transitions=warmup, warmup_reward_events=int(np.count_nonzero(rewards[:warmup] > 0)),
                reward_events=len(hits), reward_events_per_100000_transitions=float(100000 * len(hits) / total),
                training_goal_counts=counts.tolist(), training_goal_coverage=int(np.count_nonzero(counts)),
                curve=dict(transitions=grid, reward_events=cumulative_rewards.tolist(),
                           reward_discovered=[int(len(hits) > 0 and first(hits[0]) <= t) for t in grid],
                           goals_discovered=cumulative_goals), reward_frequency_bins=frequencies)


def initial_collection(config, cache):
    from .envs.drac_paper import PaperPointMaze
    path = cache / f"{config['maze']}-s{config['seed']}-n{config['num_envs']}.npz"
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    n, seed = config['num_envs'], config['seed']
    actions = np.random.default_rng(seed).uniform(-1., 1., (n, 2)).astype(np.float32)
    env = PaperPointMaze(config['maze'])
    values = {k: [] for k in ('observations', 'next_observations', 'rewards', 'terminals')}
    try:
        goals = env.goal_positions.copy()
        for i, action in enumerate(actions):
            obs, _ = env.reset(seed=seed + 100004 * i)
            nxt, reward, terminal, truncated, _ = env.step(action)
            values['observations'].append(obs)
            values['next_observations'].append(nxt)
            values['rewards'].append(reward)
            values['terminals'].append(float(terminal or truncated))
    finally:
        env.close()
    data = {k: np.asarray(v, dtype=np.float32) for k, v in values.items()}
    data.update(actions=actions, goals=goals)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def analyze_replay(run, cache):
    config = read(run / 'config.json')
    done = read(run / 'completed.json')
    assert done['completed'] and done['updates'] == config['expected_updates']
    prefix = initial_collection(config, cache)
    path = run / 'checkpoints/final-replay.npz'
    with np.load(path) as data:
        size, position = int(data['size']), int(data['position'])
        total, n = config['steps'], config['num_envs']
        lost = total - size
        assert 0 <= lost < n and position == total % config['replay_capacity']
        keys = dict(observations='states' if 'states' in data else 'observations',
                    next_observations='next_states' if 'next_states' in data else 'next_observations',
                    actions='actions', rewards='rewards')
        arrays = {key: data[value].reshape((size, -1)) if key != 'rewards' else data[value].reshape(size)
                  for key, value in keys.items()}
        # Confirm recovery against every retained row from the same initial batch.
        for key, value in arrays.items():
            np.testing.assert_array_equal(value[lost:n], prefix[key][lost:n])
        order = np.concatenate((np.arange(position, size), np.arange(position))) if lost else np.arange(size)
        rewards = np.concatenate((prefix['rewards'][:lost], arrays['rewards'][order]))
        states = np.concatenate((prefix['next_observations'][:lost], arrays['next_observations'][order]))
        assert len(rewards) == total
        result = discovery_metrics(rewards, states, prefix['goals'], n, config['warmup'])
    result.update(prefix_reconstructed_transitions=lost, prefix_validation='all retained initial-batch observation/action/reward/next-observation rows match exactly',
                  replay_path=str(path), replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  elapsed_seconds_shared_gpu=done.get('elapsed_seconds'),
                  timing_note='Shared-GPU campaign elapsed time; not an isolated efficiency benchmark')
    return result


def evaluations(run):
    path = run / 'evaluations.jsonl'
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mode = row['primary_evaluation_mode']
        result.append(dict(steps=row['steps'], updates=row['updates'], mode=mode,
                           normal=goal_metrics(row[mode]), obstacle=goal_metrics(row['obstacle_' + mode])))
    return result


def verify_training_source(run):
    hashes = read(run / 'source-hashes.json')
    checked = 0
    for name, expected in hashes.items():
        # Report-only additions are allowed; every recorded inference/env file must match.
        if name.startswith(('pointmaze_rebuttal/agents/', 'pointmaze_rebuttal/ibolt_core/',
                            'pointmaze_rebuttal/envs/', 'pointmaze_rebuttal/profiles/')) or name == 'pointmaze_rebuttal/evaluation.py':
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Inference source differs from training: ' + name)
            checked += 1
    assert checked > 10
    return checked


def evaluate_full(run, output, device, smoke=False):
    import flax.serialization
    import jax
    from .agents.ibolt import Agent
    from .evaluation import evaluate
    assert_work_branch()
    assert_output_boundary(output)
    config = read(run / 'config.json')
    assert config['method'] == 'ibolt'
    source_files = verify_training_source(run)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'evaluations').mkdir()
    agent = Agent(config, output, device=device)

    class FullPolicy:
        evaluation_mode = 'full_policy'

        def act(self, obs, mode):
            assert mode == self.evaluation_mode
            return agent.act(obs, mode='train')

        def evaluation_rng(self, seed):
            return agent.evaluation_rng(seed)

    policy = FullPolicy()
    checkpoints = sorted((run / 'checkpoints').glob('*.msgpack'))
    if smoke:
        checkpoints = checkpoints[-1:]
    try:
        write(output / 'config.json', dict(training_run=str(run), training_config=config,
            evaluation_policy='full_policy', latent='fresh Gaussian each step', conditional='sampled box-truncated Gaussian',
            training_source_files_verified=source_files, smoke=smoke, backend=jax.default_backend()))
        for checkpoint in checkpoints:
            step = int(checkpoint.stem)
            payload = checkpoint.read_bytes()
            restored = flax.serialization.from_bytes(
                dict(actor=agent.model.policy.actor_state, critic=agent.model.policy.qf_state), payload)
            agent.model.policy.actor_state = restored['actor']
            agent.model.policy.qf_state = restored['critic']
            updates = int(restored['actor'].step)
            assert updates == (step - config['warmup']) // config['num_envs'] * config['updates_per_collect']
            # Repeated RNG contexts must reproduce samples, and the conditional draw must be active.
            obs = np.zeros((32, 4), np.float32)
            with agent.evaluation_rng(900000 + config['seed']):
                full = agent.act(obs, 'train')
            with agent.evaluation_rng(900000 + config['seed']):
                repeated = agent.act(obs, 'train')
            with agent.evaluation_rng(900000 + config['seed']):
                centers = agent.act(obs, 'mu_only')
            np.testing.assert_array_equal(full, repeated)
            assert np.max(np.abs(full - centers)) > 1e-5
            episodes = 5 if smoke else config['final_episodes'] if step == config['steps'] else config['eval_episodes']
            record = dict(steps=step, updates=updates, method='ibolt', primary_evaluation_mode='full_policy',
                          checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(payload).hexdigest())
            for obstacle, offset in ((False, 17000), (True, 27000)):
                key = 'obstacle_full_policy' if obstacle else 'full_policy'
                record[key] = evaluate(policy, 'pm_' + config['maze'], config['seed'] + offset + step,
                    episodes, 'full_policy', output / 'evaluations' / f'{step:09d}_{key}.npz', obstacle)
            assert int(agent.model.policy.actor_state.step) == updates
            assert checkpoint.read_bytes() == payload
            write(output / 'evaluations' / f'{step:09d}_summary.json', record)
            with (output / 'evaluations.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(dict(seed=config['seed'], steps=step, success=record['full_policy']['success'],
                                  coverage=record['full_policy']['reachable_goals'])), flush=True)
        write(output / 'completed.json', dict(completed=True, checkpoints=len(checkpoints), smoke=smoke, evaluation_only=True))
    finally:
        agent.close()


def aggregate(entries, expected=5):
    rows = []
    for method in METHODS:
        selected = [r for r in entries if r['method'] == method]
        training = [r['training'] for r in selected if r.get('training')]
        for variant in (('native', 'full_policy') if method == 'ibolt' else ('native',)):
            final, histories = [], []
            for entry in selected:
                points = entry.get('evaluations' if variant == 'native' else 'full_policy_evaluations', [])
                histories.append({p['steps']: p for p in points})
                last = next((p for p in points if p['steps'] == entry['total_steps']), None)
                if last:
                    final.append(last)
            common = sorted(set.intersection(*(set(h) for h in histories))) if len(histories) == expected else []
            curves = [dict(transitions=t, success=stats([h[t]['normal']['success'] for h in histories]),
                           coverage=stats([h[t]['normal']['reachable_goals'] for h in histories])) for t in common]
            valid_first = [t['first_reward_at_collection_end'] for t in training if t['reward_discovered']]
            row = dict(method=method, evaluation_variant=variant, expected_seeds=expected,
                training_completed=len(training), final_evaluations=len(final),
                complete=len(training) == expected and len(final) == expected,
                reward_discovery_seeds=sum(t['reward_discovered'] for t in training),
                reward_not_discovered_seeds=sum(not t['reward_discovered'] for t in training),
                all_training_goals_discovered_seeds=sum(t['all_goals_discovered'] for t in training),
                first_reward_conditional_on_discovery=stats(valid_first),
                first_reward_median_conditional_on_discovery=statistics.median(valid_first) if valid_first else None,
                undiscovered_seed_handling='kept in discovery fractions; first-event times are right-censored at total budget',
                warmup_reward_events=stats([t['warmup_reward_events'] for t in training]),
                reward_events_per_100000_transitions=stats([t['reward_events_per_100000_transitions'] for t in training]),
                success=stats([v['normal']['success'] for v in final]),
                coverage=stats([v['normal']['reachable_goals'] for v in final]),
                normalized_goal_entropy_conditional_on_success=stats([v['normal']['normalized_goal_entropy'] for v in final]),
                return_=stats([v['normal']['mean_return'] for v in final]),
                obstacle_success=stats([v['obstacle']['success'] for v in final]),
                obstacle_sr5=stats([v['obstacle']['sr5_obstacle'] for v in final]), evaluation_curve=curves)
            if len(training) == expected:
                grids = [t['curve']['transitions'] for t in training]
                assert all(g == grids[0] for g in grids)
                grid = sorted(set(grids[0]) | {time for v in training
                    for time in v['first_goal_reward_at_collection_end'] if time is not None})
                row['discovery_curve'] = [dict(transitions=t,
                    discovered_seed_fraction=sum(v['first_reward_at_collection_end'] is not None and
                        v['first_reward_at_collection_end'] <= t for v in training) / expected,
                    training_goal_coverage=stats([sum(time is not None and time <= t
                        for time in v['first_goal_reward_at_collection_end']) for v in training]))
                    for t in grid]
            rows.append(row)
    return rows


def render(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    for name, fields in [('discovery', ['discovered_seed_fraction', 'training_goal_coverage']),
                         ('evaluation', ['success', 'coverage'])]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for row in report['methods']:
            key = 'discovery_curve' if name == 'discovery' else 'evaluation_curve'
            if name == 'discovery' and row['evaluation_variant'] != 'native':
                continue
            points = row.get(key, [])
            if not points:
                continue
            x = np.array([p['transitions'] for p in points]) / 1e6
            label = row['method'] + (' (full)' if row['evaluation_variant'] == 'full_policy' else ' (mu only)' if row['method'] == 'ibolt' else '')
            for ax, field in zip(axes, fields):
                values = [p[field] for p in points]
                y = np.array([v['mean'] if isinstance(v, dict) else v for v in values])
                line, = ax.plot(x, y, label=label, drawstyle='steps-post' if name == 'discovery' else 'default')
                if isinstance(values[0], dict):
                    sd = np.array([v['sample_std'] or 0. for v in values])
                    ax.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=.12,
                                    step='post' if name == 'discovery' else None)
                ax.set(xlabel='Environment transitions (millions)', ylabel=field.replace('_', ' '))
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        if name == 'discovery':
            axes[0].set(xlim=(0, .2), ylim=(-.02, 1.02), title='Reward discovery in the first 200K transitions')
        fig.suptitle(report['maze'].title() + ': five independent training seeds; sample SD')
        fig.savefig(figures / f'{name}.svg')
        fig.savefig(figures / f'{name}.png', dpi=160)
        plt.close(fig)
    labels, values = [], []
    for entry in report['per_seed']:
        for variant in ('native', 'full_policy') if entry['method'] == 'ibolt' else ('native',):
            history = entry.get('evaluations' if variant == 'native' else 'full_policy_evaluations', [])
            final = next((p for p in history if p['steps'] == entry['total_steps']), None)
            if final:
                labels.append(f"{entry['method']} {variant} s{entry['seed']}")
                values.append(final['normal']['goal_proportions_of_successes'])
    if values:
        fig, ax = plt.subplots(figsize=(8, max(3, .24 * len(values))), constrained_layout=True)
        heat = ax.imshow(values, vmin=0, vmax=1, aspect='auto', cmap='viridis')
        ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel='Goal ID (zero based)',
               xticks=range(len(values[0])), title='Goal proportions among successful final evaluation episodes')
        ax.tick_params(axis='y', labelsize=7)
        fig.colorbar(heat, ax=ax)
        fig.savefig(figures / 'goal-distributions.svg')
        fig.savefig(figures / 'goal-distributions.png', dpi=160)
        plt.close(fig)
    def cell(value):
        if value['mean'] is None:
            return 'pending'
        sd = value['sample_std']
        return f"{value['mean']:.4g} ± {sd:.3g} (n={value['n']})" if sd is not None else f"{value['mean']:.4g} (n={value['n']})"
    table = []
    for row in report['methods']:
        table.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in [
            row['method'] + ' / ' + row['evaluation_variant'], row['training_completed'],
            f"{row['reward_discovery_seeds']}/{row['training_completed']} analyzed",
            row['first_reward_median_conditional_on_discovery'], cell(row['success']), cell(row['coverage']),
            cell(row['normalized_goal_entropy_conditional_on_success']), row['complete']]) + '</tr>')
    status = html.escape(report['status'])
    page = f'''<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="120">
<title>PointMaze reviewer evidence: {report['maze']}</title>
<style>body{{font:15px system-ui;max-width:1200px;margin:36px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}img{{max-width:100%}}.note{{background:#eef3f8;padding:14px}}</style>
<h1>PointMaze {report['maze']}: reviewer evidence</h1><p>Status: {status}. Native training runs analyzed: {report['training_analyzed']}/35. Full-policy iBOLT reevaluations: {report['full_policy_completed']}/5.</p>
<p class="note">Partial results are labeled with n. Statistics use independent training seeds and sample SD (ddof=1); episodes are not independent training runs. Discovery uses real training rewards. First-event times are rounded to the end of a 256-environment collection. Undiscovered runs remain censored. Coverage requires at least one visit; the goal distribution is also shown. iBOLT mu-only and full-policy results remain separate. Shared-GPU elapsed times are not efficiency benchmarks.</p>
<table><tr><th>Method / evaluation</th><th>Training n/5</th><th>Reward found</th><th>First reward median transitions, discovered runs</th><th>Final success</th><th>Final coverage</th><th>Normalized goal entropy</th><th>Complete</th></tr>{''.join(table)}</table>
<p><a href="summary.json">Summary and curves JSON</a> · <a href="per-seed.json">Per-seed discovery times and evaluation distributions</a> · <a href="status.json">Job status and errors</a></p>
<img src="figures/discovery.svg"><img src="figures/evaluation.svg"><img src="figures/goal-distributions.svg">
<p>Comparison protocol: Medium/Hard, seeds 0–4, 1,000,192 transitions, warmup 8192, vector envs 256, UTD=1, batch 256; native optimizer schedules retained. Intermediate evaluations: 200 episodes per condition; final: 500. No learning updates occur in the reporting or full-policy evaluator.</p></html>'''
    tmp = root / 'report.html.tmp'
    tmp.write_text(page)
    tmp.replace(root / 'report.html')


def monitor(config_path, once=False):
    assert_work_branch()
    config = read(config_path)
    root = Path(config['output'])
    assert_output_boundary(root)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'status.json'
    state = read(status_path) if status_path.exists() else dict(full_policy={}, analysis_errors={})
    child = None
    child_seed = None
    while True:
        entries = []
        errors = {}
        for job in config['jobs']:
            run = Path(job['output'])
            cfg = read(run / 'config.json')
            entry = dict(method=job['method'], seed=job['seed'], maze=config['maze'], training_run=str(run),
                         total_steps=cfg['steps'], training=None, evaluations=evaluations(run))
            cached = root / 'runs' / f"{job['method']}-s{job['seed']}.json"
            if cached.exists():
                entry['training'] = read(cached)
            elif (run / 'completed.json').exists():
                try:
                    verify_training_source(run)
                    entry['training'] = analyze_replay(run, root / 'prefix-cache')
                    write(cached, entry['training'])
                    print(json.dumps(dict(event='replay_analyzed', method=job['method'], seed=job['seed'])), flush=True)
                except Exception as error:
                    errors[f"{job['method']}-s{job['seed']}"] = repr(error)
            if (run / 'failure.json').exists():
                entry['training_failure'] = read(run / 'failure.json')
                errors[f"{job['method']}-s{job['seed']}"] = entry['training_failure']
            if job['method'] == 'ibolt':
                folder = root / 'full-policy' / f"ibolt-s{job['seed']}"
                entry['full_policy_evaluations'] = evaluations(folder)
                if (folder / 'completed.json').exists():
                    state['full_policy'][str(job['seed'])] = 'completed'
            entries.append(entry)
        if child is not None and child.poll() is not None:
            folder = root / 'full-policy' / f'ibolt-s{child_seed}'
            state['full_policy'][str(child_seed)] = 'completed' if child.returncode == 0 and (folder / 'completed.json').exists() else 'failed'
            child = None
        if child is None and not once:
            for job in config['jobs']:
                seed = job['seed']
                if job['method'] != 'ibolt' or str(seed) in state['full_policy']:
                    continue
                destination = root / 'full-policy' / f'ibolt-s{seed}'
                if destination.exists():
                    state['full_policy'][str(seed)] = 'interrupted_output_preserved'
                    continue
                free = subprocess.check_output(['nvidia-smi', '-i', str(config['evaluation_gpu']),
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
                if int(free.strip()) < 2000:
                    break
                log_path = root / 'logs' / f'full-policy-s{seed}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config['evaluation_gpu']), XLA_PYTHON_CLIENT_PREALLOCATE='false',
                           MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                env.pop('JAX_PLATFORMS', None)
                command = ['taskset', '-c', ','.join(map(str, config['evaluation_cpus'])), config['python'], '-u',
                           '-m', 'pointmaze_rebuttal.reporting', '--evaluate-ibolt', job['output'], '--output', str(destination), '--device', 'cuda']
                with log_path.open('x') as log:
                    child = subprocess.Popen(command, cwd=config['source'], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                child_seed = seed
                state['full_policy'][str(seed)] = 'running'
                state['full_policy_pid'] = child.pid
                break
        methods = aggregate(entries)
        analyzed = sum(e['training'] is not None for e in entries)
        full_done = sum(v == 'completed' for v in state['full_policy'].values())
        complete = analyzed == 35 and full_done == 5 and not errors
        state.update(updated_unix=time.time(), pid=os.getpid(), training_analyzed=analyzed, full_policy_completed=full_done,
                     analysis_errors=errors, status='completed' if complete else 'collecting', complete=complete)
        report = dict(maze=config['maze'], status=state['status'], updated_unix=state['updated_unix'],
                      training_analyzed=analyzed, full_policy_completed=full_done, methods=methods, per_seed=entries)
        write(root / 'per-seed.json', entries)
        write(root / 'summary.json', {k: v for k, v in report.items() if k != 'per_seed'})
        write(status_path, state)
        render(root, report)
        if complete:
            write(root / 'completed.json', dict(completed=True, training_runs=35, full_policy_runs=5, finished_unix=time.time()))
        if complete or once:
            return
        time.sleep(config.get('poll_seconds', 120))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monitor', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--evaluate-ibolt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.monitor:
        monitor(args.monitor, args.once)
    elif args.evaluate_ibolt and args.output:
        evaluate_full(args.evaluate_ibolt, args.output, args.device, args.smoke)
    else:
        parser.error('Specify --monitor or --evaluate-ibolt with --output')


if __name__ == '__main__':
    main()
