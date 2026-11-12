"""Copy campaign checkpoints in bounded, independent per-run transfers.

Host 1 has its own archive service. Historical archives and hash-verified pruning
remain handled by the existing collectors. This service never deletes a file.
"""
from central_collector import *
import re
import signal

MANAGED_HOSTS = (3, 4, 5)


def write_status(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def transfer(n, job, dest):
    name = job['id']
    assert re.fullmatch(r'[a-z0-9-]+(?:p[a-z0-9-]+)*', name), name
    run = dest / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    args = ['rsync', '-a', '--partial', '--timeout=120', '--stats',
            '--exclude=*.tmp', '--exclude=.git/', '--exclude=jax-cache/',
            '--exclude=mpl-cache/', '--include=*/',
            '--include=**/checkpoints/***', '--exclude=*',
            '-e', shlex.join(ssh(n)),
            'heechan@' + HOSTS[n][0] + ':' + NEW + '/runs/' + name + '/', str(run) + '/']
    started = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        # Only this transfer's isolated process group; no learner or controller.
        try:
            import os
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
        return dict(id=name, passed=False, seconds=time.time()-started, error='transfer timeout after 600s')
    size = re.search(r'Total transferred file size: ([\d,]+) bytes', stdout)
    return dict(id=name, passed=proc.returncode in (0, 24), code=proc.returncode,
                seconds=time.time()-started,
                transferred_bytes=int(size[1].replace(',', '')) if size else 0,
                error=stderr[-800:] if proc.returncode not in (0, 24) else None)


def cycle(n):
    dest = ROOT / f'hosts/host{n}/workspace' / NEW
    plan = json.loads((dest / 'plan.json').read_text())
    jobs = [j for j in plan['jobs'] if j['status'] in ('running', 'completed')
            and j['phase'] != 'smoke']
    previous_file = ROOT / f'checkpoint-sync-host{n}.json'
    previous = json.loads(previous_file.read_text()) if previous_file.exists() else {}
    prior_times = {r['id']: r.get('finished', 0) for r in previous.get('runs', [])}
    # Oldest served run first; new jobs cannot be stuck behind a growing tree.
    jobs.sort(key=lambda j: (prior_times.get(j['id'], 0), j['id']))
    out = dict(at=time.time(), started=time.time(), host=n, in_progress=True,
               passed=None, jobs=len(jobs), finished_jobs=0, transferred_bytes=0,
               runs=[], last_success_at=previous.get('last_success_at'))
    write_status(previous_file, out)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4 if n == 4 else 2) as executor:
        futures = {executor.submit(transfer, n, j, dest): j['id'] for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(id=futures[future], passed=False, error=repr(exc))
            result['finished'] = time.time()
            out['runs'].append(result)
            out.update(at=time.time(), finished_jobs=len(out['runs']))
            out['transferred_bytes'] += result.get('transferred_bytes', 0)
            write_status(previous_file, out)
    out.update(at=time.time(), in_progress=False, passed=all(r['passed'] for r in out['runs']))
    if out['passed']:
        out['last_success_at'] = time.time()
    write_status(previous_file, out)


def loop(n):
    while True:
        try:
            cycle(n)
        except Exception as exc:
            write_status(ROOT / f'checkpoint-sync-host{n}.json',
                         dict(at=time.time(), host=n, passed=False, error=repr(exc)))
        time.sleep(45)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MANAGED_HOSTS)) as executor:
        list(executor.map(loop, MANAGED_HOSTS))
