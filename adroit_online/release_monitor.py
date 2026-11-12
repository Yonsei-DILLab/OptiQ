"""Hourly read-only monitor; exits when both instances pass the completion gate.

Stopping remains a separate action after off-instance archive verification.
The latest user instruction forbids destruction/return; retain instance disks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

SERVERS = [(60326900, '175.155.64.205', 19159), (60327673, '179.255.127.236', 31020)]


def ready(reports):
    return (len(reports) == 2 and
            {r.get('instance_id') for r in reports} == {60326900, 60327673} and
            all(r.get('ready_for_final_backup') and r.get('completed') == 32 and
                r.get('total') == 32 and not r.get('failures') and
                not r.get('gpu_processes') for r in reports))


def snapshot(server, code):
    instance, host, port = server
    command = ['ssh', '-S', 'none', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
               '-p', str(port), 'heejoonorm@' + host, 'python3 -']
    try:
        p = subprocess.run(command, input=code, capture_output=True, text=True, timeout=90)
        assert p.returncode == 0, p.stderr[-3000:]
        result = json.loads(p.stdout)
        assert result['instance_id'] == instance
        return result
    except Exception as error:
        return dict(instance_id=instance, error=str(error), ready_for_final_backup=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interval', type=int, default=3600)
    p.add_argument('--once', action='store_true')
    a = p.parse_args(); assert a.interval >= 60
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = Path(__file__).with_name('release_probe.py').read_text()
    print(json.dumps(dict(event='monitor_started', pid=os.getpid(),
                          interval_seconds=a.interval, instances=[s[0] for s in SERVERS])), flush=True)
    while True:
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(lambda server: snapshot(server, code), SERVERS))
        record = dict(observed_unix=time.time(), reports=reports,
                      all_complete=ready(reports), next_check_unix=time.time() + a.interval)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        text = json.dumps(record, indent=2) + '\n'
        (a.output / (stamp + '.json')).write_text(text)
        temporary = a.output / 'latest.tmp'; temporary.write_text(text)
        temporary.replace(a.output / 'latest.json')
        print(json.dumps(dict(event='all_complete' if record['all_complete'] else 'hourly_check',
                              observed_unix=record['observed_unix'], reports=[
                                  {k: r.get(k) for k in ('instance_id', 'completed', 'total',
                                                       'failures', 'error')} for r in reports])), flush=True)
        if record['all_complete'] or a.once:
            return
        if any(r.get('failures') for r in reports):
            raise RuntimeError('Experiment failure requires review; release is blocked')
        deadline = start + a.interval
        while time.monotonic() < deadline:
            time.sleep(min(60, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
