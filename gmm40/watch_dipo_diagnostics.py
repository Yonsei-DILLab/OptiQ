"""Run isolated DIPO replay probes only after evaluations are durably published."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .evaluation import atomic_json
from .target import RESULTS
from .repair_gif_timing import repair_published_run


def refresh_report(output):
    records = [json.loads(path.read_text()) for path in sorted(output.glob('update_*.json'))]
    lines = ['# DIPO independent checkpoint probes', '',
             f'Updated (UTC): {datetime.now(timezone.utc).isoformat()}', '',
             'Frozen-checkpoint replay probes, not losses recorded by the live optimizer.',
             'Each probe uses a separate fixed RNG; no main-training state is modified.', '',
             '| Updates | Probe samples | Twin TD MSE sum | Denoising loss | Mean Q | Mean TD target | Refined action saturation |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        name = f"update_{row['updates']:07d}.json"
        lines.append(f"| [{row['updates']:,}]({name}) | {row['probe_samples']:,} | {row['critic_loss_sum_of_twin_mse']:.4f} | {row['actor_denoising_loss']:.5f} | {row['mean_q']:.3f} | {row['mean_td_target']:.3f} | {row['refined_action_saturation_fraction']:.2%} |")
    lines.extend(['', 'TD error measures consistency with a sampled learned target, not true-Q accuracy.',
                  'Saturation counts action coordinates with absolute value above .99.',
                  'Interpret these probes together with rollout return and terminal distribution metrics.'])
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return records


def poll(name, samples):
    folder = RESULTS/name
    config = json.loads((folder/'config.json').read_text())
    status = json.loads((folder/'status.json').read_text())
    if config.get('method') != 'dipo' or config.get('Q') != 'learned, not oracle':
        raise ValueError('Expected a DIPO navigation run')
    if status.get('status') == 'failed':
        raise RuntimeError(f'Source training failed: {status}')
    output = RESULTS/'diagnostics/navigation_dipo'/name
    output.mkdir(parents=True, exist_ok=True)
    published = []
    source = folder/'metrics.jsonl'
    if source.exists():
        for line in source.read_text().splitlines():
            try:
                published.append(json.loads(line)['updates'])
            except json.JSONDecodeError:
                # A final appended line can be observed before its write completes.
                break
    source_hash = hashlib.sha256(Path(__file__).with_name('diagnose_navigation_dipo.py').read_bytes()).hexdigest()
    for step in sorted(set(published)-{0}):
        path = output/f'update_{step:07d}.json'
        prior = json.loads(path.read_text()) if path.exists() else {}
        if (prior.get('probe_samples') == samples and prior.get('input_checkpoint_unchanged')
                and prior.get('parameter_finiteness_passed')
                and prior.get('diagnostic_source_sha256') == source_hash):
            continue
        checkpoint = folder/'checkpoints'/f'update_{step:07d}.bin'
        atomic_json(output/'watch_status.json', dict(status='probing',pid=os.getpid(),updates=step))
        completed = subprocess.run([sys.executable,'-m','gmm40.diagnose_navigation_dipo',
                                    str(checkpoint),'--samples',str(samples)],
                                   text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f'Probe failed for {checkpoint}: {completed.stdout}\n{completed.stderr}')
        print(completed.stdout, flush=True)
    records = refresh_report(output)
    timing = repair_published_run(folder)
    budget = config['actor_updates']
    expected = set(range(10000,budget+1,10000)) | {budget}
    done = status.get('status') == 'completed' and timing['deferred'] == 0
    probed = {row['updates'] for row in records}
    if done and not expected.issubset(probed):
        raise RuntimeError(f'Completed source run missing probes: {sorted(expected-probed)}')
    result = dict(status='completed' if done else 'waiting_for_next_published_checkpoint',
                  pid=os.getpid(), source_run=name, source_updates=status.get('updates'),
                  probed_updates=sorted(probed), budget=budget,
                  gif_timing=timing,
                  timestamp=datetime.now(timezone.utc).isoformat())
    atomic_json(output/'watch_status.json',result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',default='navigation_dipo_native_seed0_100k')
    parser.add_argument('--samples',type=int,default=4096)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    output = RESULTS/'diagnostics/navigation_dipo'/args.name
    output.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(event='watch_started',pid=os.getpid(),source_run=args.name,
                          probe_samples=args.samples)),flush=True)
    try:
        while True:
            result = poll(args.name,args.samples)
            if args.once or result['status'] == 'completed':
                print(json.dumps(result),flush=True)
                return
            time.sleep(20)
    except Exception as exc:
        atomic_json(output/'watch_status.json',dict(status='failed',pid=os.getpid(),error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
