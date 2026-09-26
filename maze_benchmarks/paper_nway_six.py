"""Read-only iBOLT 4/8/16-Way trajectory and dense learned-Q plate."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence, Type, Optional

import flax.linen as nn
from flax.serialization import msgpack_restore
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'models/utils.py': ['GELU'],
    'models/critic.py': ['Critic', 'VectorCritic'],
    'analysis_tools/experiments/20260920_truncated_mll/optiq_dime/policy.py':
        ['kernel_init', 'SemiImplicitActor'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(source, names):
    return [n for n in ast.parse(source).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]


def original_networks(commit):
    # Load the actual frozen Flax definitions without importing training services.
    # BatchRenorm is unreachable: this report explicitly rejects batch norm.
    scope = dict(nn=nn, jnp=jnp, jax=jax, Sequence=Sequence, Type=Type,
                 Optional=Optional, BatchRenorm=nn.BatchNorm)
    provenance = {}
    for path, names in SOURCES.items():
        frozen = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT, text=True)
        nodes = definitions(frozen, names)
        current = definitions((ROOT/path).read_text(), names)
        module = ast.Module(body=nodes, type_ignores=[])
        assert ast.dump(module) == ast.dump(ast.Module(body=current, type_ignores=[]))
        exec(compile(module, f'{commit}:{path}', 'exec'), scope)
        provenance[path] = hashlib.sha256(ast.dump(module).encode()).hexdigest()
    return scope, provenance


def infer(folder, old_probe, proof, output, grid=101, samples=128):
    cfg = json.loads((folder/'config.json').read_text())
    assert cfg['temperature'] in (1, 3, 5, 10) and cfg['method'] == 'optiq'
    checkpoint = folder/'checkpoints'/f"policy_{cfg['steps']:09d}.msgpack"
    assert sha(checkpoint) == proof['checkpoint_sha256']
    assert sha(folder/'config.json') == proof['config_sha256']
    assert sha(old_probe) == proof['sha256'][old_probe.name]
    net, code_hashes = original_networks(cfg['source_commit'])
    alg = cfg['agent']['alg']; ac = alg['actor']; qc = alg['critic']
    assert not alg['optimizer']['bn'] and not qc['use_layer_norm']
    assert not qc['dropout_rate'] and qc['n_atoms'] == 1 and qc['activation'] == 'gelu'
    actor = net['SemiImplicitActor'](action_dim=2, hidden_dims=ac['hidden_dims'],
        log_std_min=ac['log_std_min'], log_std_max=ac['log_std_max'],
        initial_log_std=ac['initial_log_std'],
        mean_output_init_scale=ac['mean_output_init_scale'],
        log_std_output_init_scale=ac.get('log_std_output_init_scale', 0.),
        mean_latent_skip_scale=ac.get('mean_latent_skip_scale', 0.))
    critic = net['VectorCritic'](net_arch=qc['hs'], activation_fn=net['GELU'],
        batch_norm_momentum=.99, n_critics=qc['n_critics'], n_atoms=1)
    saved = msgpack_restore(checkpoint.read_bytes())
    ap = jax.tree.map(jnp.asarray, saved['actor']['params'])
    cp = jax.tree.map(jnp.asarray, saved['critic']['params'])

    @jax.jit
    def q(s, a):
        return critic.apply({'params': cp}, s, a, train=False).mean(axis=0).reshape(-1)

    # Check the original twin-critic mean against the archived GPU probe.
    with np.load(old_probe) as old:
        xx, yy = np.meshgrid(old['x'], old['y'])
        obs = np.repeat(np.stack([xx.ravel(), yy.ravel()], -1), old['actions'].shape[2], axis=0)
        actual = np.asarray(q(jnp.asarray(obs, jnp.float32), jnp.asarray(old['actions'].reshape(-1, 2))))
        error = np.abs(actual - old['q'].reshape(-1))
        # Archived GPU inference and current CPU kernels are not bit-identical.
        # Bound both worst-case and RMS error relative to the full Q range;
        # keep the measured absolute discrepancy in the report provenance.
        q_range = float(np.ptp(old['q']))
        if error.max() > .001*q_range or np.sqrt(np.mean(error**2)) > .0002*q_range:
            raise ValueError('Restored Q does not match preserved inference')

    # A common normal latent bank removes state-to-state Monte Carlo jitter;
    # every state retains the same N(0,I) prior. No sigma noise or Q filtering.
    latent = jnp.asarray(np.random.default_rng(271828).normal(size=(samples, 2)), jnp.float32)

    @jax.jit
    def values(states):
        obs = jnp.repeat(states, samples, axis=0)
        z = jnp.tile(latent, (states.shape[0], 1))
        mu, _ = actor.apply({'params': ap}, obs, z)
        vals = q(obs, mu).reshape(-1, samples)
        return vals.mean(axis=1), vals.std(axis=1) / np.sqrt(samples)

    coords = np.linspace(-7., 7., grid, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    states = np.stack([xx.ravel(), yy.ravel()], -1)
    means, errors = [], []
    for start in range(0, len(states), 64):
        part = states[start:start+64]
        padded = np.pad(part, ((0, 64-len(part)), (0, 0)), mode='edge')
        mean, se = values(jnp.asarray(padded))
        means.extend(np.asarray(mean)[:len(part)])
        errors.extend(np.asarray(se)[:len(part)])
    np.savez_compressed(output, x=coords, y=coords,
                        q=np.asarray(means).reshape(grid, grid),
                        mc_standard_error=np.asarray(errors).reshape(grid, grid), latent=np.asarray(latent))
    assert sha(checkpoint) == proof['checkpoint_sha256']
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        training_source=cfg['source_commit'], frozen_network_ast_sha256=code_hashes,
        restored_probe_max_abs_error=float(error.max()),
        restored_probe_rmse=float(np.sqrt(np.mean(error**2))),
        restored_probe_q_range=q_range, dense_grid=grid,
        normal_latent_samples=samples, latent_seed=271828, learner_updates=0,
        q_definition='E_z [(Q1(s,mu(s,z))+Q2(s,mu(s,z)))/2]',
        conditional_sigma=False, common_random_numbers=True, spatial_smoothing=False,
        training_steps=cfg['steps'], temperature=cfg['temperature'],
        actor_updates=int(saved['actor']['step']),
        batch_size=cfg['batch_size'], output_sha256=sha(output))


def goals(n):
    if n == 4:
        return np.array([[5., 0], [-5., 0], [0, 5.], [0, -5.]])
    angle = 2*np.pi*np.arange(n)/n
    return 6*np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def render(root, out, records, temperature=1, paper_only=False):
    plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'path', 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(23.8, 3.5 if paper_only else 4.4), facecolor='white')
    for i, n in enumerate((4, 8, 16)):
        name = f'{n}way-optiq-t{temperature:g}-s0'; folder = root/'runs'/name
        cfg = json.loads((folder/'config.json').read_text())
        raw = folder/'evaluations'/f"{cfg['steps']:09d}_mu_only.npz"
        with np.load(raw) as z:
            assert str(z['mode']) == 'mu_only'
            tracks, ids = z['xy'], z['goal_ids']
        proof = json.loads((root/'posthoc_mu'/name/'proof.json').read_text())
        assert sha(raw) == proof['preserved_mu_sha256']
        counts = np.bincount(ids[ids >= 0], minlength=n)
        progress = json.loads((folder/'progress.json').read_text())
        assert progress['status'] == 'complete' and progress['steps'] == cfg['steps']
        assert counts.tolist() == progress['latest_evaluation']['mu_only']['goals']
        xy = goals(n)
        ax = fig.add_subplot(1, 6, 2*i+1)
        c = np.linspace(-7.7, 7.7, 401); xx, yy = np.meshgrid(c, c)
        # Goal-proximity reference, not a learned density or the reward function.
        # Nearest Gaussian gives separate goal contours even for closely spaced goals.
        d2 = ((np.stack([xx, yy], -1)[..., None, :] - xy)**2).sum(-1).min(-1)
        reference = np.exp(-d2/(2*1.35**2))
        ax.contour(xx, yy, reference, levels=np.linspace(.08, .96, 12),
                   cmap='viridis', linewidths=.65, alpha=.85)
        # Keep every recorded trajectory, including rare modes and failures.
        for path in tracks:
            path = path[np.isfinite(path).all(axis=-1)]
            ax.plot(path[:, 0], path[:, 1], color='#dd1e27', lw=.8, alpha=.18)
        # Thin arrowheads only. Add the first recorded rollout for each observed
        # outcome so rare routes have directional arrows too; no path is invented.
        indices = np.unique(np.r_[np.linspace(0, len(tracks)-1, 32, dtype=int),
                                 [np.flatnonzero(ids == goal)[0] for goal in np.unique(ids)]])
        for index in indices:
            path = tracks[index]; path = path[np.isfinite(path).all(axis=-1)]
            positions = np.arange(0, len(path)-1, 3)
            if len(positions):
                delta = path[positions+1] - path[positions]
                ax.quiver(path[positions, 0], path[positions, 1], delta[:, 0], delta[:, 1],
                          color='#dd1e27', angles='xy', scale_units='xy', scale=1,
                          width=.0048, headwidth=3.5, headlength=4.5, alpha=.95, zorder=5)
        ax.scatter(xy[:, 0], xy[:, 1], s=17, color='#dd1e27', edgecolors='white', linewidths=.5, zorder=6)
        ax.scatter([0], [0], s=14, color='#202020', zorder=7)
        ax.set(xlim=(-7.7, 7.7), ylim=(-7.7, 7.7), aspect='equal',
               xticks=[-6, -3, 0, 3, 6], yticks=[-6, -3, 0, 3, 6], xlabel='x', ylabel='y')
        if not paper_only:
            ax.set_title(f'{n}-Way · trajectories', fontsize=13, pad=12)
        ax.tick_params(labelsize=8)
        if not paper_only:
            ax.text(.5, -.18, f'Goals reached: {np.count_nonzero(counts)}/{n}',
                    transform=ax.transAxes, ha='center', fontsize=11)
        ax = fig.add_subplot(1, 6, 2*i+2, projection='3d')
        with np.load(out/f'{n}way_dense_q.npz') as q:
            x, y = np.meshgrid(q['x'], q['y']); value = q['q']
        ax.plot_surface(x, y, value, rcount=len(x), ccount=len(y), cmap='viridis',
                        linewidth=0, edgecolor='none', antialiased=True, shade=True)
        ax.set(xlabel='x', ylabel='y', zlabel='Learned Q', xticks=[-6, 0, 6], yticks=[-6, 0, 6])
        if not paper_only:
            ax.set_title(f'{n}-Way · Q', fontsize=13, pad=12)
        ax.view_init(elev=29, azim=-55); ax.set_box_aspect((1, 1, .75))
        ax.tick_params(labelsize=8, pad=0)
        records[name].update(rollout_sha256=sha(raw), episodes=len(ids),
                             shown_trajectories=len(tracks),
                             trajectory_style=dict(color='#dd1e27',linewidth=.8,alpha=.18,
                                                   arrow_width=.0048,arrow_alpha=.95,
                                                   same_style_for_every_trajectory=True),
                             arrowhead_rollout_indices=indices.tolist(),
                             goal_counts=counts.tolist(), failures=int(sum(ids < 0)))
    if paper_only:
        fig.subplots_adjust(left=.025, right=.98, bottom=.06, top=.99, wspace=.30)
    else:
        fig.suptitle(f'iBOLT · T = {temperature:g} · random-z, μ-only', fontsize=16, fontweight='bold', y=.99)
        fig.subplots_adjust(left=.025, right=.98, bottom=.2, top=.85, wspace=.30)
        fig.text(.5, .035,
            'All 1,024 saved rollouts; arrowheads thinned for clarity. Contours: goal-proximity reference. '
            'Q: mean of twin critics, averaged over 128 Gaussian latents per state.',
            ha='center', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'ibolt_4_8_16way_T{temperature:g}_1x6.{ext}', dpi=240, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--temperature', type=int, choices=(1, 3, 5, 10), default=1)
    p.add_argument('--paper-only', action='store_true', help='Remove titles, footnotes and coverage text; keep axis labels')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True); records = {}
    for n in (4, 8, 16):
        name = f'{n}way-optiq-t{a.temperature}-s0'; folder = a.root/'runs'/name
        proof = json.loads((a.root/'posthoc_mu'/name/'proof.json').read_text())
        old = a.root/'posthoc_mu'/name/f"{proof['step']:09d}_probe_mu_only.npz"
        records[name] = infer(folder, old, proof, a.output/f'{n}way_dense_q.npz')
        print(f'{n}-Way dense Q verified and saved', flush=True)
    render(a.root, a.output, records, a.temperature, paper_only=a.paper_only)
    manifest = dict(runs=records, reporting_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope='Read-only final-checkpoint inference; no retraining, no checkpoint mutation',
        temperature=a.temperature, paper_only=a.paper_only,
        caveat=('All selected runs have 62000 updates and 1000192 transitions.' if a.temperature != 1 else
                '4-Way has 62000 updates; 8/16-Way have 998976. These are existing T1 results, not a matched-training-budget comparison.'),
        reference_contours='exp(-nearest_goal_distance_squared/(2*1.35^2)); illustrative goal-proximity, not reward or learned density',
        outputs={ext:sha(a.output/f'ibolt_4_8_16way_T{a.temperature}_1x6.{ext}') for ext in ('png', 'svg')})
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
