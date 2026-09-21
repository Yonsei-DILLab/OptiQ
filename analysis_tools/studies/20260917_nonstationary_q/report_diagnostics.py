"""Plot saved observations only. No fitting, extra policy sampling, or smoothing."""
import numpy as np
import matplotlib.pyplot as plt


def panels(root, ts, figdir, methods, sizes, labels, colors, stages,
           densities, records, finish):
    images = []
    for stage in stages:
        for n, m in sizes:
            selected = [t for t in ts if t['stage'] == stage and
                        t['n'] == n and t['seed'] == 0]
            available = [(t, densities(root, t)) for t in selected]
            available = [(t, ds) for t, ds in available if len(ds) > 2]
            if not available:
                continue
            # Predetermined time slices, never choose snapshots by their scores.
            times = ([20000, 20020, 25020, 35000] if stage == 'mass' else
                     [20000, 22000, 27000, 35000] if stage == 'split' else
                     [1000, 10000, 20000, 35000])
            fig, axes = plt.subplots(2 * len(available), len(times),
                                     figsize=(19, 4.4 * len(available)), squeeze=False,
                                     gridspec_kw={'height_ratios': [2, 1] * len(available)})
            for i, (t, ds) in enumerate(available):
                for j, step in enumerate(times):
                    ax = axes[2 * i, j]; teacher_ax = axes[2 * i + 1, j]
                    ax.set_title(f'{labels[t["method"]]} | {step:,}', fontsize=9)
                    if step not in ds:
                        ax.text(.5, .5, 'Not yet available', transform=ax.transAxes,
                                ha='center')
                        teacher_ax.set_visible(False)
                        continue
                    with np.load(ds[step]) as z:
                        edges = z['edges']; width = np.diff(edges)
                        centers = (edges[:-1] + edges[1:]) / 2
                        ax.plot(centers, z['target_bin_mass'] / width, '--',
                                color='#243248', label='Target')
                        teacher_ax.stairs(z['proposal_counts'] / z['proposal_actions'].size / width,
                                  edges, color='#999999', lw=.7, alpha=.6,
                                  label=f'Proposal ({m:,} candidates)')
                        teacher_ax.stairs(z['teacher_bin_mass'] / width, edges,
                                  color='#bc8b16', lw=.8, alpha=.8,
                                  label='Weighted teacher')
                        teacher_ax.plot(centers, z['target_bin_mass'] / width,
                                        '--', color='#243248', lw=.8)
                        ax.stairs(z['density'], edges, color=colors[list(methods).index(t['method'])],
                                  lw=.9, label='Actor (32,768 samples)')
                    if i == 0 and j == 0:
                        ax.legend(fontsize=7)
                        teacher_ax.legend(fontsize=7)
                    ax.set_xlim(-1, 1); teacher_ax.set_xlim(-1, 1)
                    if i == len(available) - 1:
                        teacher_ax.set_xlabel('Action')
                    if j == 0:
                        ax.set_ylabel('Actor / target density')
                        teacher_ax.set_ylabel('Teacher density')
            name = f'40_teacher_actor_{stage}_{n}x{m}.png'
            finish(fig, figdir / name)
            images.append((f'{stage}, {n}×{m}: target / proposal / teacher / actor (seed 0)', name))

            if stage in ('replay', 'closed'):
                fig, axes = plt.subplots(len(available), 2,
                                         figsize=(13, 2.7 * len(available)), squeeze=False)
                for i, (t, ds) in enumerate(available):
                    for step in times:
                        if step not in ds:
                            continue
                        with np.load(ds[step]) as z:
                            axes[i, 0].plot(z['grid'], z['q'], label=f'update {step:,}')
                            axes[i, 1].plot(z['grid'], z['pdf'], label=f'update {step:,}')
                    for j, ylabel in enumerate(['Live critic Q(0,a)', 'Target density']):
                        axes[i, j].set(title=labels[t['method']], xlabel='Action', ylabel=ylabel)
                    if axes[i, 0].lines:
                        axes[i, 0].legend(fontsize=7)
                name = f'43_learned_q_{stage}_{n}x{m}.png'
                finish(fig, figdir / name)
                images.append((f'{stage}, {n}×{m}: actual learned Q and its target (seed 0)', name))

            # Follow exactly the same saved evaluation latent indices across time.
            fig, axes = plt.subplots(len(available), 2,
                                     figsize=(13, 2.7 * len(available)), squeeze=False)
            for i, (t, ds) in enumerate(available):
                steps = sorted(ds)
                mu, sigma, latent = [], [], None
                ix = np.linspace(0, 32767, 16, dtype=int)
                for step in steps:
                    with np.load(ds[step]) as z:
                        this_z = z['z'][ix]
                        if latent is None:
                            latent = this_z.copy()
                        else:
                            assert np.array_equal(latent, this_z), 'Fixed latent mismatch'
                        mu.append(z['mu'][ix]); sigma.append(np.exp(z['log_sigma'][ix]))
                for j, arr in enumerate([np.tanh(mu), np.asarray(sigma)]):
                    axes[i, j].plot(steps, arr, lw=.8, alpha=.65)
                    axes[i, j].set(xlabel='Actor update', title=labels[t['method']],
                                   ylabel='Conditional median tanh(mu)' if j == 0 else
                                   'Pre-tanh sigma')
                axes[i, 0].set_ylim(-1, 1)
            name = f'41_fixed_latents_{stage}_{n}x{m}.png'
            finish(fig, figdir / name)
            images.append((f'{stage}, {n}×{m}: fixed latent mean / sigma (seed 0)', name))

            fig, axes = plt.subplots(2, 3, figsize=(17, 8))
            fields = [('binned_tv', 'Actor histogram TV'),
                      ('teacher_binned_tv', 'Teacher histogram TV'),
                      ('backup_bias', 'Actor backup - target backup'),
                      ('sigma_mean', 'Mean sigma (pre-tanh)'),
                      ('between_mu_variance', 'Between-mu variance'),
                      ('usage_ess', 'Assignment usage ESS / N')]
            for method, color in zip(methods, colors):
                rows = [records(root, t) for t in ts if t['stage'] == stage and
                        t['n'] == n and t['method'] == method]
                rows = [r for r in rows if r]
                if not rows:
                    continue
                # Intersection avoids changes in seed composition along the mean curve.
                common = sorted(set.intersection(*[set(r['step'] for r in rr) for rr in rows]))
                maps = [{r['step']: r for r in rr} for rr in rows]
                for ax, (field, title) in zip(axes.flat, fields):
                    values = np.array([[r[s][field] for s in common] for r in maps])
                    if field == 'usage_ess':
                        values /= n
                    mean = values.mean(0)
                    sd = values.std(0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                    ax.plot(common, mean, color=color, label=f'{labels[method]} (seeds={len(rows)})')
                    ax.fill_between(common, mean - sd, mean + sd, color=color, alpha=.12)
                    ax.set(title=title, xlabel='Actor update')
            axes[0, 0].legend(fontsize=7)
            name = f'42_metrics_{stage}_{n}x{m}.png'
            finish(fig, figdir / name)
            images.append((f'{stage}, {n}×{m}: tracking and component metrics (seed mean ± SD)', name))
    return images
