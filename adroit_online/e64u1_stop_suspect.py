"""Stop only the verified GPU1 peer with reproduced XLA red-zone errors."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from .e64u1_retry import ORIGINAL
from .priority_controller import alive, process_ticks, verify_adopted
from .run import atomic, source_provenance

NAME = 'hammer-ibolt-s4-e64-u1-t005'


def main(root):
    proof = source_provenance()
    receipt = root / 'gpu1-reviewed-stop.json'; assert not receipt.exists()
    q = json.loads((root / 'queue-state.json').read_text())
    state = q['jobs'][NAME]
    assert q['source_commit'] == ORIGINAL and '1' in q['quarantined']
    assert state['status'] == 'running' and state['gpu'] == 1 and alive(state)
    source = root / ('source-' + ORIGINAL)
    verify_adopted(root, NAME, state, source)
    log = (root / 'process-logs' / (NAME + '-running.log')).read_text()
    assert 'Red zone modified' in log
    rows = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
            rows[int(p.name)] = int(stat[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    selected = {state['pid']}
    while True:
        more = {pid for pid, parent in rows.items() if parent in selected}
        if more <= selected: break
        selected.update(more)
    targets = []
    for pid in sorted(selected):
        p = Path('/proc') / str(pid); ticks = process_ticks(pid)
        if ticks is None: continue
        assert p.stat().st_uid == os.getuid() and (p / 'cwd').resolve() == source
        targets.append(dict(pid=pid, start_ticks=ticks,
                            command=(p / 'cmdline').read_bytes().decode().split(chr(0))))
    record = dict(operation_commit=proof['commit'], source_commit=ORIGINAL, run_id=NAME,
                  reason='GPU1 peer red-zone errors after adjacent illegal-address recurrence',
                  targets=targets, progress=json.loads((root / 'runs' / NAME / 'progress.json').read_text()),
                  red_zone_errors=log.count('Red zone modified'), requested_unix=time.time())
    atomic(receipt, record)
    for target in targets:
        if process_ticks(target['pid']) == target['start_ticks']:
            os.kill(target['pid'], signal.SIGTERM)
    for _ in range(20):
        live = [t for t in targets if process_ticks(t['pid']) == t['start_ticks']]
        if not live: break
        time.sleep(1)
    assert not live, 'Review remaining owned processes; do not kill an unverified group'
    record.update(stopped=True, stopped_unix=time.time())
    atomic(receipt, record)
    print(json.dumps(record))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
