"""Read-only numerical/provenance audit; copies approved inputs to a new study."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    keys = ['n', 'm', 'batch', 'temperature', 'learning_rate', 'hidden_dims',
            'log_std_min', 'log_std_max', 'initial_log_std', 'mean_output_init_scale',
            'teacher_std_floor', 'action_bound', 'target_width', 'target_centers',
            'target_kind']
    records = []
    for case in cfg['cases']:
        target_keys = (['target_widths', 'target_masses'] if case['target_kind'] == 'gmm'
                       else ['shapes', 'shape_masses', 'core_intervals', 'expected_modes'])
        for seed in cfg['seeds']:
            pair = []
            for method in ['forward', 'reverse']:
                src = a.archive / 'campaign/runtime/confirm' / case['id'] / f'{method}_s{seed}'
                old = json.loads((src / 'RUN.json').read_text())
                done = json.loads((src / 'COMPLETE.json').read_text())
                metrics = json.loads((src / 'metrics_100000.json').read_text())
                expected = dict(cfg, **case)
                mismatches = {k: [old['config'].get(k), expected[k]] for k in keys + target_keys
                              if old['config'].get(k) != expected[k]}
                assert not mismatches, (src, mismatches)
                assert done['step'] == 100000 and metrics['step'] == 100000
                assert old['method'] == method and old['seed'] == seed
                assert old['L'] == (1048576 if method == 'reverse' else 0)
                assert metrics['sample_count'] == 1048576
                assert done['source_commit'] == old['source_commit']
                ck_hash = sha(src / 'checkpoint.msgpack')
                assert old.get('parent_checkpoint_sha256', ck_hash) == ck_hash
                rec = dict(target=case['id'], method=method, seed=seed, n=128, m=128,
                           L=old['L'], origin=str(src), source_commit=old['source_commit'],
                           initial_parameter_sha256=old['initial_parameter_sha256'],
                           checkpoint_sha256=ck_hash, config_match=True,
                           archived_complete_step=done['step'],
                           restore_validation='pending runtime validation',
                           original_run_sha256=sha(src / 'RUN.json'))
                dst = a.root / 'reused' / case['id'] / f'{method}_s{seed}'
                dst.mkdir(parents=True, exist_ok=True)
                for name in ['RUN.json', 'COMPLETE.json', 'checkpoint.msgpack']:
                    shutil.copy2(src / name, dst / name)
                for metric in src.glob('metrics_*.json'):
                    shutil.copy2(metric, dst / metric.name)
                (dst / 'ORIGIN.json').write_text(json.dumps(rec, indent=2) + '\n')
                records.append(rec)
                pair.append(rec)
            assert pair[0]['initial_parameter_sha256'] == pair[1]['initial_parameter_sha256']
    result = dict(status='metadata_and_hash_audit_passed', entries=records,
                  total_entries=108, baseline_candidates=len(records), new_training_runs=96,
                  final_reuse_gate='fresh initialization and 100K checkpoint restore on execution runtime')
    (a.root / 'REUSE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'entries'}))


if __name__ == '__main__':
    main()
