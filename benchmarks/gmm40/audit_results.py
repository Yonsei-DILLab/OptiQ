"""Join tuning results by sample hash; never combine metrics across candidates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import wandb

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/gmm40_tuning')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for file in [os.environ.get('OPTIQ_ENV_FILE'), ROOT/'.env', ROOT.parent/'.env']:
        if file:
            load_dotenv(file, override=False)
    exports = {}
    for path in sorted(args.root.glob('exports/*/summary.json')):
        export = read(path)
        sample_hash = export['samples_sha256']
        if sha256(path.parent/'samples.npy') != sample_hash:
            raise ValueError('Sample file changed: '+str(path.parent))
        if sha256(Path(export['checkpoint'])) != export['checkpoint_sha256']:
            raise ValueError('Checkpoint changed: '+export['checkpoint'])
        if sample_hash in exports:
            raise ValueError('Duplicate sample hashes need an explicit report alias')
        cfg = export['parent_config']
        exports[sample_hash] = dict(label=path.parent.name, export=str(path),
            samples_sha256=sample_hash, checkpoint_sha256=export['checkpoint_sha256'],
            actor_update=export['actor_update'], training_seed=cfg['seed'],
            K=cfg['num_policy_samples'], R=cfg['proposals_per_policy_sample'],
            hidden_dims=cfg['hidden_dims'], temperature=cfg['temperature'],
            density_beta=cfg['density_beta'], proposal_std=cfg['proposal_std'],
            direct_native_unbounded=(export['direct_generator_samples']
                and cfg['coordinate_scale'] == 1 and cfg['unbounded_actions']),
            sample_reports=[], cfm_evaluations=[])
    for path in sorted(args.root.glob('paper_eval_*/summary.json')):
        report = read(path)
        key = report['config']['samples_sha256']
        if key in exports:
            label = report['config']['label']
            records = [r for r in report['records'] if r['method'] == label]
            if len(records) != 1:
                raise ValueError('Ambiguous sampler row in '+str(path))
            exports[key]['sample_reports'].append(dict(path=str(path),
                config=report['config'], metrics=records[0], wandb_url=report['wandb_url']))
    posts = [dict(path=str(p), **read(p)) for p in args.root.glob('post_cfm*/summary.json')]
    cfm_configs = set(args.root.glob('cfm_hp*/config.json')) | set(args.root.glob('cfm_confirmation*/config.json'))
    for path in sorted(cfm_configs):
        cfg = read(path)
        if cfg.get('gt_control') or cfg.get('samples_sha256') not in exports:
            continue
        row = dict(path=str(path.parent), config=cfg, finished=False, posts=[])
        summary = path.parent/'summary.json'
        if summary.exists():
            row['finished'] = True
            row['raw_final'] = read(summary)
        history = path.parent/'history.jsonl'
        if history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            row['latest_validation'] = rows[-1]
            row['best_validation'] = min(rows, key=lambda r:r['validation_nll'])
        for post in posts:
            if post['parent_config'].get('wandb_url') == cfg['wandb_url']:
                if post['checkpoint_sha256'] != sha256(path.parent/'best_cfm.pt'):
                    raise ValueError('Post-evaluation used a different CFM checkpoint')
                row['posts'].append(post)
        exports[cfg['samples_sha256']]['cfm_evaluations'].append(row)
    protocol = read(ROOT/'benchmarks/gmm40/goal_protocol.json')
    all_metrics_required = protocol.get('user_revision', {}).get('all_paper_metrics_required', True)
    for candidate in exports.values():
        candidate['declared_confirmation'] = any(
            sample['config'].get('confirmation', False) for sample in candidate['sample_reports'])
    confirmation_candidates = [c['label'] for c in exports.values() if c['declared_confirmation']]
    tuning_only = not confirmation_candidates
    report = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        tuning_only=tuning_only, overall_goal_achieved=False,
        confirmation_candidates=confirmation_candidates,
        all_paper_metrics_required=all_metrics_required, scope_completion_checked=False,
        note='This joins measured evidence; it is not a completion verifier. '
             + ('The user removed the all-paper-metrics winning requirement. '
                'The legacy overall_goal_achieved field is not a completion decision for the revised scope. '
                if not all_metrics_required else '')
             + 'Independent-seed confirmation is declared per sample evaluation; '
             'recipe, seed coverage and completion still require a separate review. '
             'NLL means final held-out CFM test NLL; validation and KDE diagnostics do not replace it. '
             'ESS16 is the author-disclosed original paper count. ESS1000 is supplemental. '
             'Histogram-range and ODE-tolerance ambiguities remain disclosed in the protocol. '
             'Training-time comparison requires its own hardware/ancestry audit.',
        protocol=protocol, candidates=list(exports.values()))
    lines = ['# GMM40 evidence by fixed candidate', '', report['note'], '',
        'All metric joins use the exact100k direct-sample SHA256; checkpoint and sample hashes were reread. '
        'The table shows raw final CFM results. ESS16 post-evaluations use the same fixed CFM checkpoint at tolerance1e-3. '
        'All post-evaluation values, including stricter tolerances and changes to ESS1000, are retained in audit.json.', '',
        '| Candidate | Seed | log p | Modes | W2 | TV | Final NLL | ESS16 | Raw ESS1000 | Raw log Z LB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '—' if value is None else f'{value:.4f}'
    for candidate in exports.values():
        samples = candidate['sample_reports'] or [None]
        cfms = candidate['cfm_evaluations'] or [None]
        # Preserve every reference/evaluator variant rather than choosing a good one.
        for sample in samples:
            for cfm in cfms:
                sm = sample['metrics'] if sample else {}
                cm = cfm.get('raw_final', {}) if cfm else {}
                post_ess = []
                for post in cfm['posts'] if cfm else []:
                    if post.get('reference_seed') is not None:
                        continue
                    for result in post['results']:
                        if result['tolerance'] == .001 and 'normalized_ess_batch16_mean' in result:
                            post_ess.append(result['normalized_ess_batch16_mean'])
                ess16 = ', '.join(fmt(x) for x in post_ess) or '—'
                lines.append('| '+ ' | '.join([candidate['label'],str(candidate['training_seed']),
                    fmt(sm.get('mean_log_prob')),str(sm.get('modes_covered','—')),
                    fmt(sm.get('w2_mean')),fmt(sm.get('spatial_tv_mean')),
                    fmt(cm.get('test_nll')),ess16,fmt(cm.get('normalized_ess')),
                    fmt(cm.get('log_z_lower_bound'))])+' |')
    lines += ['', 'Missing entries mean unmeasured or unfinished. They are not passes. '
        'Repeated sample metrics describe resampling variability, not independent training-seed uncertainty.', '']
    run = wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'), mode='online', dir=str(args.output),
        group='gmm40-idem-protocol', job_type='evidence-audit', name='GMM40-fixed-candidate-audit',
        config=dict(tuning_only=tuning_only, source_sha256=sha256(Path(__file__))))
    report['wandb_url'] = run.url
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'analysis.md').write_text('\n'.join(lines))
    artifact=wandb.Artifact('gmm40-evidence-audit-'+run.id,type='gmm40-evaluation')
    for name in ['audit.json','analysis.md']:
        artifact.add_file(str(args.output/name))
    run.log_artifact(artifact)
    run.summary.update(dict(candidates=len(exports),overall_goal_achieved=False,tuning_only=tuning_only,
                           confirmation_candidates=len(confirmation_candidates),
                           all_paper_metrics_required=all_metrics_required,scope_completion_checked=False))
    run.finish()
    print(json.dumps(dict(output=str(args.output),candidates=len(exports),wandb_url=report['wandb_url'])))


if __name__ == '__main__':
    main()
