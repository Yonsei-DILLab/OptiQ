"""Write a self-contained Markdown report with metrics derived from raw outputs."""
from pathlib import Path


def write_report(out, provenance):
    density_rows = []
    score_rows = []
    probe_rows = []
    names = {'t00_reference': 'Three Gaussian modes', 'n00_spike_ramp': 'Spike + ramp'}
    for case, name in names.items():
        for method in ['forward', 'reverse']:
            metrics = provenance['metrics'][f'{case}/{method}']
            density_rows.append(f'| {name} | {method.capitalize()} KL | '
                f'{metrics["TV"]["mean"]:.4f} ± {metrics["TV"]["sd"]:.4f} | '
                f'{metrics["W1"]["mean"]:.4f} ± {metrics["W1"]["sd"]:.4f} | '
                f'{metrics["missing_modes"]["seeds"]} |')
        for run in provenance['score_runs']:
            if run['case'] != case:
                continue
            summary = run['summary']
            record = next(x for x in summary['results'] if x['L'] == 2**20)
            record10 = next(x for x in summary['results'] if x['L'] == 2**10)
            score_rows.append(f'| {name} | {run["seed"]} | '
                f'{record10["policy_relative_RMSE_mean"]*100:.3f}% | '
                f'{record["policy_relative_RMSE_mean"]*100:.3f}% | '
                f'{summary["reference_policy_SE_RMS"]:.6f} |')
            probe_rows.append(f'| {name} | {run["seed"]} | '
                f'{record["target_probe_max_abs_bias"]:.4f} | '
                f'{summary["precision"]["policy_max_abs_difference"]:.2e} |')
    quantile_rows = []
    for row in provenance['score_quantiles']:
        if row['seed'] == 0:
            quantile_rows.append(f'| {names[row["case"]]} | {row["quantile"]:.0%} | '
                f'{row["action"]:.6f} | {row["score_L20_mean"]:.6f} | '
                f'{row["reference"]:.6f} | {row["score_L20_sd"]:.6f} |')
    template = Path(__file__).with_name('report_template.md').read_text()
    for key, value in {
        'DENSITY_TABLE': '\n'.join(density_rows),
        'SCORE_TABLE': '\n'.join(score_rows),
        'QUANTILE_TABLE': '\n'.join(quantile_rows),
        'PROBE_TABLE': '\n'.join(probe_rows),
        'RENDER_COMMIT': provenance['render_commit'],
        'MEASUREMENT_COMMIT': provenance['measurement_commit'],
    }.items():
        template = template.replace('{{' + key + '}}', value)
    assert '{{' not in template
    (out / 'report.md').write_text(template)
    (out / 'captions.md').write_text(r'''# Paper captions

## Forward versus reverse KL

**Forward- and reverse-KL fitting of multimodal Boltzmann targets.** Panels (a,b) use a three-Gaussian target; panels (c,d) use a non-Gaussian spike-and-ramp target. Black dashed curves show the target density. Blue and orange curves are the means of four seed-wise, 512-bin histograms, each obtained from $2^{20}$ policy samples after 100,000 updates; light fills show the learned densities and faint traces show individual seeds. Both methods use the same semi-implicit actor, $N=M=128$, and 32 independent groups per update. Reverse KL uses an independent density bank of $L=2^{20}$ latent samples per group to estimate its action score; Forward KL does not use this auxiliary bank. Vertical scales differ between panels. In these selected examples, Forward KL recovers all target modes, while Reverse KL misses modes in all four seeds.

## Score convergence

**Monte Carlo convergence of the reverse-policy action score.** The final 100,000-update Reverse-KL actor is frozen. Each panel fixes one action at the 10th, 50th, or 90th percentile of its sampled policy distribution (seed 0). Curves show the mean over 16 independent latent-bank realizations, and blue shading denotes their 10th–90th percentile range. Prefixes are nested across $L$ within each realization. Dotted horizontal lines give the mean score from four additional, independent banks of $2^{24}$ latent samples each; these are empirical references, not exact scores. The dashed vertical line marks the training choice $L=2^{20}$. The zoomed version uses independent vertical limits; the common-y version uses a shared vertical range within each target. All-seed results and off-policy probes are reported separately. The three-Gaussian target is shown in the top row and spike-and-ramp in the bottom row of the combined figure.
''')
