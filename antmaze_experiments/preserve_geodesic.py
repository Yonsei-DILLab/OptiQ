"""Cancel only Euclidean jobs and adopt unchanged geodesic processes/queue.

Original manifests and frozen training source stay immutable. Handoff and
controller provenance are separate sidecars; the existing learner is not resumed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def identity(pid):
    p = psutil.Process(pid)
    return dict(pid=pid, create_time=p.create_time(), cmdline=p.cmdline())


def alive(record):
    try:
        p = psutil.Process(record['pid'])
        return p.create_time() == record['create_time'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def prepare(root):
    receipt = root / 'euclidean-cancellation.json'
    assert not receipt.exists(), 'Handoff already prepared'
    state = json.loads((root/'status.json').read_text())
    controller = psutil.Process(state['controller_pid'])
    assert str(root) in controller.cmdline() and '--job' not in controller.cmdline()
    # Freeze only scheduling, then snapshot; learners continue uninterrupted.
    controller.send_signal(signal.SIGSTOP)
    time.sleep(.2)
    state = json.loads((root/'status.json').read_text())
    kept = [x for x in state['running'] if 'geodesic' in x['reward_profile']]
    removed = [x for x in state['running'] if 'euclidean' in x['reward_profile']]
    assert len(kept)+len(removed) == len(state['running'])
    records = {x['id']: identity(x['pid']) for x in kept}
    pending_cancelled = [x for x in state['pending'] if 'euclidean' in x]
    pending_kept = [x for x in state['pending'] if 'geodesic' in x]
    audit = dict(time=time.time(), original_state=state, retained_processes=records,
                 cancelled_running=[x['id'] for x in removed],
                 cancelled_pending=pending_cancelled, retained_pending=pending_kept,
                 reason='User replaced Euclidean runs with geodesic B=0 / no step penalty',
                 prepared=False)
    write(receipt, audit)
    # Keep the scheduler present while cancelling descendants. Orphaning a
    # process group while a member is stopped can deliver SIGHUP to keepers.
    victims=[]
    for entry in removed:
        try:
            parent=psutil.Process(entry['pid'])
            assert str(root) in parent.cmdline() and entry['id'] in parent.cmdline()
            tree=parent.children(recursive=True)+[parent]
            victims.extend(tree)
            for p in tree:
                try: p.terminate(); p.send_signal(signal.SIGCONT)
                except psutil.NoSuchProcess: pass
        except psutil.NoSuchProcess: pass
    _,remaining=psutil.wait_procs(victims, timeout=15)
    for p in remaining:
        try: p.kill()
        except psutil.NoSuchProcess: pass
    _,remaining=psutil.wait_procs(remaining,timeout=5)
    assert all(p.status()==psutil.STATUS_ZOMBIE for p in remaining), remaining
    # No stopped job parents remain. Only the controller PID is killed.
    controller.kill()
    for key in audit['cancelled_running']+pending_cancelled:
        path=root/'jobs'/(key+'.json')
        old=json.loads(path.read_text()) if path.exists() else dict(id=key)
        write(root/'jobs'/(key+'-before-cancellation.json'),old)
        write(path,dict(**{k:v for k,v in old.items() if k!='status'},status='cancelled',cancelled_at=time.time()))
    assert all(alive(r) for r in records.values()), 'Retained job exited during handoff; inspect before adopting'
    audit.update(prepared=True,finished=time.time())
    write(receipt,audit)
    state.update(controller_pid=None,running=kept,pending=pending_kept,
                 cancelled=audit['cancelled_running']+pending_cancelled, scheduler_handoff_pending=True)
    write(root/'status.json',state)
    print(json.dumps(dict(cancelled=state['cancelled'],preserved=list(records),pending=pending_kept)))


def monitor(root):
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=json.loads((root/'euclidean-cancellation.json').read_text());assert audit['prepared']
    assert not (root/'geodesic-adoption.json').exists(), 'Do not restart adoption automatically'
    manifest=json.loads((root/'manifest.json').read_text())
    source=Path(manifest['source'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==manifest['source_commit']
    code=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    write(root/'geodesic-adoption.json',dict(time=time.time(),controller_source=str(code),controller_commit=sha,
        training_source=str(source),training_commit=manifest['source_commit'],retained_processes=audit['retained_processes']))
    entries={x['id']:x for x in manifest['jobs']}
    state=json.loads((root/'status.json').read_text())
    pending=[entries[k] for k in state['pending']]
    live={x['gpu']:(audit['retained_processes'][x['id']],entries[x['id']],None) for x in state['running']}
    completed=state['completed'];failed=state['failed'];cancelled=state['cancelled']
    while live or pending:
        for gpu,(record,entry,proc) in list(live.items()):
            if proc is not None: proc.poll()
            if alive(record):continue
            job=json.loads((root/'jobs'/(entry['id']+'.json')).read_text())
            result_path=root/'runs'/entry['id']/'result.json'
            if job.get('status')=='completed' and result_path.exists():
                result=json.loads(result_path.read_text())
                assert result['completed'] and result['source_commit']==manifest['source_commit']
                completed.append(entry['id'])
            else:
                failed.append(dict(id=entry['id'],gpu=gpu,reason='Adopted process exited without verified completion'))
                write(root/'failure.json',dict(failed=failed,pending_held=True,time=time.time()))
            del live[gpu]
        if not failed:
            for gpu in range(4):
                if gpu in live or not pending:continue
                with Path(f'/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock').open('a') as probe:
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                entry=pending.pop(0)
                cmd=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-direct-gmm',
                     '/home/heechan/.venv-ddiffpg-native/bin/python','-m','antmaze_experiments.controller',
                     '--root',str(root),'--job',entry['id'],'--gpu',str(gpu)]
                with (root/'logs'/(entry['id']+'-job.log')).open('x') as log:
                    proc=subprocess.Popen(cmd,cwd=source,env=dict(os.environ,OPTIQ_SOURCE_DIR=str(source),CAMPAIGN_GPU=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
                live[gpu]=(identity(proc.pid),entry,proc)
        write(root/'status.json',dict(source_commit=manifest['source_commit'],controller_commit=sha,
            controller_pid=os.getpid(),time=time.time(),pending=[x['id'] for x in pending],
            running=[dict(**entry,gpu=gpu,pid=record['pid']) for gpu,(record,entry,_) in live.items()],
            completed=completed,failed=failed,cancelled=cancelled,pending_held=bool(failed)))
        if failed and not live:return 1
        time.sleep(2)
    write(root/'geodesic-result.json',dict(completed=True,source_commit=manifest['source_commit'],
        controller_commit=sha,cancelled=cancelled,results={key:json.loads((root/'runs'/key/'result.json').read_text()) for key in completed}))
    return 0


def register(root):
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources')/sha
    audit=json.loads((root/'euclidean-cancellation.json').read_text());assert audit['prepared']
    service=root.name+'-geodesic-only'
    conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(service+'.conf')
    assert not conf.exists()
    conf.write_text(f'''[program:{service}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.preserve_geodesic --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{root.name}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/geodesic-controller.log
stderr_logfile={root}/geodesic-controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    for args in (['reread'],['update',service],['start',service]):
        subprocess.run(ctl+args,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--register',action='store_true')
    args=parser.parse_args()
    assert not (args.prepare and args.register)
    raise SystemExit(prepare(args.root) if args.prepare else register(args.root) if args.register else monitor(args.root))
