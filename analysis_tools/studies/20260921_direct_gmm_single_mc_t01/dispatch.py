"""Adopt existing workers without signaling them; insert T0.1 work first."""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import os
import signal
import subprocess
import time
from ops import ROOT, ORDER, read, write, verify, environment, command


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(state=fields[0], start=fields[19])
    except FileNotFoundError:
        return None


def is_alive(record):
    p = process(record['pid'])
    return bool(p and p['start'] == record['process_start'] and p['state'] not in ('Z', 'X'))


def pending_queue(old_pending):
    return ([dict(env=e, seed=s, campaign='T01') for e in ORDER for s in range(4)]
            + [dict(**r, campaign='T025') for r in old_pending])


def free_slots(running):
    occupied = {(r['gpu'], r['slot']) for r in running}
    assert len(occupied) == len(running), 'Duplicate GPU slot'
    return [(g, s) for s in range(2) for g in range(4) if (g, s) not in occupied]


def get_old_ops(old):
    spec = importlib.util.spec_from_file_location('previous_ops', old/'ops.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spawn(job, gpu, slot, commit, old_ops):
    origin = ROOT if job['campaign'] == 'T01' else old_ops.ROOT
    run_commit = commit if job['campaign'] == 'T01' else old_ops.read(origin/'DEPLOYMENT.json')['commit']
    cmd = (command if job['campaign'] == 'T01' else old_ops.command)(job['env'], job['seed'], run_commit)
    name = f"{job['env']}_s{job['seed']}"
    out = origin/'runs'/name
    out.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[2*(gpu+4*slot):2*(gpu+4*slot)+2]
    assert len(cpus) == 2
    cmd = ['taskset', '-c', ','.join(map(str, cpus)), *cmd]
    with (out/'stdout.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=origin/'repo', env=environment(gpu),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info = process(proc.pid)
    r = dict(**job, name=name, gpu=gpu, slot=slot, pid=proc.pid, cpus=cpus,
             process_start=info['start'], command=cmd, commit=run_commit,
             controller_commit=commit, origin=str(origin), output=str(out),
             state='running', started=time.time(), validation=False, adopted=False)
    write(out/'LAUNCH.json', r)
    return r, proc


def finish(record, proc):
    r = dict(record)
    markers = list(Path(r['output']).glob('*/completed.json'))
    code = proc.poll() if proc is not None else None
    r.update(state='failed', finished=time.time(), exit_code=code)
    if len(markers) == 1:
        marker = read(markers[0])
        if marker['timesteps'] == 1000000 and code in (None, 0):
            r.update(**marker, state='complete')
    write(Path(r['output'])/'STATUS.json', r)
    return r


def self_test():
    jobs = pending_queue([dict(env='ant', seed=s) for s in range(4)])
    assert len(jobs) == 12
    assert [r['campaign'] for r in jobs] == ['T01']*8+['T025']*4
    slots = [dict(gpu=g, slot=s) for g, s in free_slots([])]
    assert free_slots(slots) == []
    assert free_slots(slots[:-1]) == [(3, 1)]
    me = process(os.getpid())
    assert is_alive(dict(pid=os.getpid(), process_start=me['start']))
    assert not is_alive(dict(pid=os.getpid(), process_start='wrong'))
    print('Scheduler priority, slot capacity and process identity checks passed.')


def main(old, pid):
    lock = (ROOT/'dispatch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = verify()
    assert read(ROOT/'PREFLIGHT.json')['passed']
    assert not (ROOT/'STATUS.json').exists(), 'Inspect existing queue; do not duplicate'
    old_ops = get_old_ops(old)
    old_launch = old_ops.verify()
    assert old_launch['commit'] == launch['numerical_base_commit']
    assert (old/'PAUSE_QUEUE').exists()
    assert Path(f'/proc/{pid}/cwd').resolve() == old.resolve()
    assert 'dispatch.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
    parent_start = process(pid)['start']
    # Stop only the scheduler. Its PTY and all numerical workers remain alive.
    os.kill(pid, signal.SIGSTOP)
    for _ in range(100):
        if process(pid)['state'] in ('T', 't'):
            break
        time.sleep(.05)
    assert process(pid)['state'] in ('T', 't')
    previous = read(old/'STATUS.json')
    write(ROOT/'PREVIOUS_STATUS.json', previous)
    running = []
    for r0 in previous['running']:
        r = dict(r0)
        info = process(r['pid'])
        r.update(campaign='T025', origin=str(old), output=str(old/'runs'/r['name']),
                 adopted=True, process_start=info['start'] if info else '')
        running.append(r)
    pending = pending_queue(previous['pending'])
    finished = [dict(r, campaign='T025', origin=str(old)) for r in previous['finished']]
    handoff = dict(time=time.time(), controller=str(ROOT), controller_pid=os.getpid(),
                   commit=launch['commit'], old_dispatcher_pid=pid,
                   old_dispatcher_start=parent_start, adopted_pids=[r['pid'] for r in running])
    write(ROOT/'HANDOFF.json', handoff)
    write(old/'QUEUE_HANDOFF.json', handoff)
    children = {}
    retired = False
    while pending or running:
        for r in running[:]:
            proc = children.get(r['pid'])
            if proc is not None:
                proc.poll()
            if not is_alive(r):
                running.remove(r)
                finished.append(finish(r, proc))
        # The old scheduler is retired only after all its original children exited.
        if not retired and not any(r.get('adopted') for r in running):
            current = process(pid)
            if current and current['start'] == parent_start:
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGCONT)
            retired = True
        for gpu, slot in free_slots(running):
            if pending and not (ROOT/'PAUSE_QUEUE').exists():
                r, proc = spawn(pending[0], gpu, slot, launch['commit'], old_ops)
                pending.pop(0)
                running.append(r)
                children[r['pid']] = proc
        status = dict(commit=launch['commit'], controller_pid=os.getpid(), time=time.time(),
                      pending=pending, running=running, finished=finished, old_dispatcher_retired=retired)
        write(ROOT/'STATUS.json', status)
        write(old/'STATUS.json', dict(commit=old_launch['commit'], time=time.time(),
            controller=str(ROOT), pending=[{k:r[k] for k in ('env','seed')} for r in pending if r['campaign']=='T025'],
            running=[r for r in running if r['campaign']=='T025'],
            finished=[r for r in finished if r['campaign']=='T025']))
        time.sleep(5)
    write(ROOT/'QUEUE_FINISHED.json', dict(commit=launch['commit'], finished=finished, time=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--old-root', type=Path)
    p.add_argument('--old-pid', type=int)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.old_root and args.old_pid
        main(args.old_root, args.old_pid)
