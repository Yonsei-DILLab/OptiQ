"""Read-only completion gate for the two explicitly authorized Vast instances.

Executed through SSH stdin. This file never stops or destroys an instance.
"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/workspace/heejoon-adroit-online-20261112')
INSTANCE_IDS = {'60326900', '60327673'}


def read(path):
    return json.loads(path.read_text())


def inspect(root=ROOT):
    instance = os.environ['CONTAINER_ID']
    assert instance in INSTANCE_IDS
    queue = read(root / 'priority-queue-state.json')['jobs']
    queue.update(read(root / 'temp001-queue-state.json')['jobs'])
    if (root / 'recovery-queue-state.json').exists():
        queue.update(read(root / 'recovery-queue-state.json')['jobs'])
    assert len(queue) == 32, 'Changed campaign scope requires review before release'
    rows = []
    for name, state in queue.items():
        folder = root / 'runs' / state.get('attempt_id', name)
        progress = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
        row = dict(run_id=name, status=state['status'], steps=progress.get('steps'),
                   last_update_unix=progress.get('updated_unix'), complete_verified=False)
        if state['status'] == 'completed':
            spec = read(folder / 'spec.json'); done = read(folder / 'completed.json')
            final = read(folder / 'evaluations' / '003000064-final.json')
            assert done['source_commit'] == spec['source_commit'] == state['source_commit']
            assert done['steps'] == spec['steps'] == progress['steps'] == 3_000_064
            assert done['updates'] == spec['expected_updates'] == progress['updates']
            assert progress['status'] == 'completed' and final['episodes'] == 1000
            assert final['kind'] == 'final' and final['step'] == 3_000_064
            assert not (folder / 'error.json').exists()
            required = ['final-replay.pkl']
            if spec['method'] == 'ibolt':
                required += ['actor_state_3000064.msgpack', 'critic_state_3000064.msgpack',
                             'auxiliary-state-003000064.pkl']
            else:
                required += ['policy-003000064.zip']
            assert all((folder / 'checkpoints' / n).stat().st_size > 0 for n in required)
            row.update(complete_verified=True, source_commit=spec['source_commit'],
                       output=str(folder), final_success=done['final_evaluation'])
        rows.append(row)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                   '--format=csv,noheader'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True).strip()
    complete = sum(r['complete_verified'] for r in rows)
    failed = [r['run_id'] for r in rows if r['status'] in ('failed', 'requires_review')]
    return dict(observed_unix=time.time(), instance_id=int(instance),
                hostname=os.uname().nodename, completed=complete, total=len(rows),
                ready_for_final_backup=complete == 32 and not apps,
                failures=failed, gpu_inventory=gpu, gpu_processes=apps, runs=rows)


if __name__ == '__main__':
    print(json.dumps(inspect()))
