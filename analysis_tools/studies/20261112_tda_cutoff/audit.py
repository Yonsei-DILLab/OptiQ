"""Frozen-result audit; no model inference or new mode/graph estimation."""
import argparse
import contextlib
import csv
import datetime
import hashlib
import importlib
import json
import os
import pickle
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy',) and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(importlib.import_module('numpy._core.multiarray'), name)
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')


def safe_load(file, **kwargs):
    return ArrayUnpickler(file, **kwargs).load()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count_modes(result, tau, cutoff):
    height_filter = np.ones(len(result['pers']), dtype=bool)
    if cutoff:
        height_filter = result['pairs'][:, 0] >= result['fmax'] - np.log(1 / cutoff)
    return int(result['n_components'] + ((result['pers'] >= tau) & height_filter).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit):
        raise ValueError('A full pre-run source commit is required')
    source = Path(__file__).resolve().parent
    inp, out = args.input.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((source / 'config.json').read_text())
    manifest = json.loads((source / 'INPUT_MANIFEST.json').read_text())
    for name, item in manifest['files'].items():
        assert sha(inp / name) == item['sha256'], name
    source_hashes = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
                     if p.is_file() and '__pycache__' not in p.parts}
    provenance = dict(source_commit=args.source_commit, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      input_archive_sha256=manifest['archive_sha256'], source_files=source_hashes,
                      python=sys.version, numpy=np.__version__, argv=sys.argv)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

    # The upstream script only reads pickles and writes its own table/figures.
    repro = out / 'original_reproduction'
    (repro / 'figs').mkdir(parents=True)
    (repro / 'bp32k').symlink_to(inp / 'bp32k', target_is_directory=True)
    previous_dir, previous_load = Path.cwd(), pickle.load
    try:
        os.chdir(repro)
        pickle.load = safe_load
        with (repro / 'stdout.txt').open('w') as log, contextlib.redirect_stdout(log):
            upstream = runpy.run_path(str(source / 'upstream' / 'cmp5_table.py'), run_name='__main__')
    finally:
        os.chdir(previous_dir)
        pickle.load = previous_load
    actual = repro / 'figs' / 'cmp5_summary.txt'
    reference = inp / 'figs' / 'cmp5_summary_reference.txt'
    byte_match = actual.read_bytes() == reference.read_bytes()
    repro_check = dict(byte_identical=byte_match, actual_sha256=sha(actual), reference_sha256=sha(reference))
    (out / 'reproduction_check.json').write_text(json.dumps(repro_check, indent=2) + '\n')
    if not byte_match:
        raise AssertionError('Upstream table differs from supplied byte reference')

    tg = np.logspace(np.log10(cfg['curve_min']), np.log10(cfg['curve_max']), cfg['curve_points'])
    actual_taus = {t: float(tg[np.argmin(abs(tg - t))]) for t in cfg['taus']}
    banks = defaultdict(list)
    inventory, checks = [], []
    pattern = re.compile(r'(ibolt|sac1m(\d+))_(hopper|humanoid)_s(\d+)\.pkl')
    for path in sorted((inp / 'bp32k').glob('*.pkl')):
        match = pattern.fullmatch(path.name)
        if not match:
            raise ValueError(path.name)
        is_ibolt = match.group(1) == 'ibolt'
        policy = 'iBOLT' if is_ibolt else 'SAC'
        seed = int(match.group(4) if is_ibolt else match.group(2))
        env, bank = match.group(3).capitalize(), int(match.group(4))
        with path.open('rb') as file:
            data = safe_load(file)
        inventory.append(dict(file=path.name, env=env, policy=policy, seed=seed, state_bank=bank,
                              states=len(data['res']), input_args=data['args']))
        for r in data['res']:
            assert r['pairs'].shape == (len(r['pers']), 2)
            assert np.isfinite(r['pairs']).all() and np.isfinite(r['pers']).all()
            assert len(r['pers']) + r['n_components'] == r['n_modes']
            assert len(r['births']) == len(r['basin_mass']) == r['n_modes']
            assert abs(float(np.sum(r['basin_mass'])) - 1) < 1e-8
            assert r['n_components'] == 1, 'Disconnected graph needs a separate counting convention'
            banks[(env, policy, seed)].append((bank, r))
            checks.append(dict(env=env, policy=policy, seed=seed, state_bank=bank, state=r['state'],
                               n_modes=r['n_modes'], n_components=r['n_components'], far=r['far'],
                               global_basin_mass=float(r['basin_mass'][np.argmax(r['births'])]),
                               minimum_persistence=float(np.min(r['pers'])) if len(r['pers']) else None))
    for key, count in cfg['expected_training_seeds'].items():
        env, policy = key.split('/')
        assert len([k for k in banks if k[:2] == (env, policy)]) == count
    for (env, policy, seed), records in banks.items():
        assert len(records) == cfg['expected_states_per_seed'][env + '/' + policy]
        assert len({(bank, r['state']) for bank, r in records}) == len(records)
    (out / 'input_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    dump_csv(out / 'integrity_by_state.csv', checks)

    states, seeds, curves = [], [], []
    grouped, curve_grouped = defaultdict(list), defaultdict(list)
    for (env, policy, seed), records in sorted(banks.items()):
        for cutoff in cfg['cutoffs']:
            for mode in ('legacy_grid', 'exact_tau'):
                for tau in cfg['taus']:
                    effective_tau = actual_taus[tau] if mode == 'legacy_grid' else tau
                    counts = np.array([count_modes(r, effective_tau, cutoff) for _, r in records])
                    baseline = np.array([count_modes(r, effective_tau, .01) for _, r in records])
                    # Removing a height cutoff must never reduce a mode count.
                    assert (counts >= baseline).all()
                    row = dict(env=env, policy=policy, seed=seed, threshold_mode=mode, cutoff=cutoff,
                               tau=tau, effective_tau=effective_tau, n_states=len(records),
                               K_mean=float(counts.mean()), fraction_multimodal=float((counts >= 2).mean()),
                               mean_added_modes=float((counts - baseline).mean()),
                               fraction_states_changed=float((counts != baseline).mean()))
                    seeds.append(row)
                    grouped[(env, policy, mode, cutoff, tau)].append(row)
                    for (bank, r), k, base in zip(records, counts, baseline):
                        states.append(dict(env=env, policy=policy, seed=seed, state_bank=bank, state=r['state'],
                                           threshold_mode=mode, cutoff=cutoff, tau=tau, effective_tau=effective_tau,
                                           K=int(k), K_at_1percent=int(base), added_modes=int(k-base)))
            for t in tg:
                mean = float(np.mean([count_modes(r, t, cutoff) for _, r in records]))
                curves.append(dict(env=env, policy=policy, seed=seed, cutoff=cutoff, tau=float(t), K_mean=mean))
                curve_grouped[(env, policy, cutoff, float(t))].append(mean)
    summaries = []
    for (env, policy, mode, cutoff, tau), rows in sorted(grouped.items()):
        row = dict(env=env, policy=policy, threshold_mode=mode, cutoff=cutoff, tau=tau,
                   effective_tau=rows[0]['effective_tau'], n_seeds=len(rows),
                   total_state_evaluations=sum(r['n_states'] for r in rows))
        for metric in ('K_mean', 'fraction_multimodal', 'mean_added_modes', 'fraction_states_changed'):
            values = np.array([r[metric] for r in rows])
            row[metric] = float(values.mean())
            row[metric + '_seed_std'] = float(values.std(ddof=1))
        summaries.append(row)
    dump_csv(out / 'summary.csv', summaries)
    dump_csv(out / 'per_seed.csv', seeds)
    dump_csv(out / 'per_state.csv', states)
    dump_csv(out / 'curves_per_seed.csv', curves)
    summary_lookup = {(r['env'],r['policy'],r['threshold_mode'],r['cutoff'],r['tau']):r for r in summaries}
    lines = ['Byte-identical upstream summary: PASS', 'Exact requested taus and legacy nearest-grid taus:',
             json.dumps(actual_taus), '', 'Exact-tau results; mean +/- sample std across training seeds:']
    for env in ('Hopper', 'Humanoid'):
        for policy in ('iBOLT', 'SAC'):
            for cutoff in cfg['cutoffs']:
                vals = [summary_lookup[(env,policy,'exact_tau',cutoff,t)] for t in cfg['taus']]
                lines.append(f'{env:8} {policy:5} cutoff={cutoff:g} n={vals[0]["n_seeds"]} ' +
                             ' '.join(f'K({v["tau"]:g})={v["K_mean"]:.6f}+/-{v["K_mean_seed_std"]:.6f}' for v in vals) +
                             f' fraction_K>=2_tau0.5={vals[0]["fraction_multimodal"]:.6f}')
    lines += ['', f'Total policy-state evaluations: {len(checks)}',
              f'States with far>0: {sum(r["far"]>0 for r in checks)}',
              f'Max far sample count: {max(r["far"] for r in checks)}',
              f'Global-maximum basin mass below 0.1%: {sum(r["global_basin_mass"]<.001 for r in checks)}']
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
    for ax, env in zip(axes, ('Hopper','Humanoid')):
        for policy, color in [('iBOLT','#2563a6'),('SAC','#c65a1e')]:
            for cutoff, linestyle, text in [(.01,'-','1%'),(0,'--','no cutoff')]:
                y = [np.mean(curve_grouped[(env,policy,cutoff,float(t))]) for t in tg]
                ax.plot(tg,y,color=color,ls=linestyle,lw=2,label=f'{policy}, {text}')
        ax.set(xscale='log',xlabel='Persistence threshold (nats)',ylabel='Mean K across training seeds',title=env)
        ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Peak-height cutoff sensitivity on the same saved modes',fontsize=12)
    fig.savefig(out/'cutoff_curves.png',dpi=180)
    fig.savefig(out/'cutoff_curves.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
