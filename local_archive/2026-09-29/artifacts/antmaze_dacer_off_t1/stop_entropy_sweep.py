"""Explicit user cancellation of the preceding entropy sweep; preserve data."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

NAME='antmaze-dense-dacer-entropy-T1-s0-20260924'
ROOT=Path('/home/heechan/optiq-experiments')/NAME
CTL=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']


def processes():
    out=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            stat=(p/'stat').read_text().split(') ',1)[1].split()
            if str(ROOT) in cmd and stat[0]!='Z':
                out.append(dict(pid=int(p.name),pgid=int(stat[2]),command=cmd))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return out


def main():
    cancellation=ROOT/'user-cancellation.json'
    assert not cancellation.exists(), 'Inspect existing cancellation before retrying'
    old=json.loads((ROOT/'status.json').read_text())
    before=processes()
    record=dict(time=time.time(),reason='User: stop all current experiments; replace with four fresh DACER OFF T1 policies.',
                previous_status=old,processes_before=before,actions=[],verified=False)
    cancellation.write_text(json.dumps(record,indent=2)+'\n')
    for name in [NAME,NAME+'-wandb-sync']:
        conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf')
        assert 'autostart=false' in conf.read_text() and 'autorestart=false' in conf.read_text()
        done=subprocess.run(CTL+['stop',name],text=True,capture_output=True,timeout=45)
        record['actions'].append(dict(service=name,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr))
    # Supervisor stopasgroup should stop all learners and vector workers.
    remain=processes()
    groups={p['pgid'] for p in before}
    for process in remain:
        assert process['pgid'] in groups and process['pgid']!=os.getpgrp()
        try:os.killpg(process['pgid'],signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+12
    while processes() and time.monotonic()<deadline:time.sleep(.5)
    remain=processes()
    if remain:
        for pgid in {p['pgid'] for p in remain}:
            assert pgid in groups and pgid!=os.getpgrp()
            try:os.killpg(pgid,signal.SIGKILL)
            except ProcessLookupError:pass
        time.sleep(1)
    remain=processes()
    assert not remain,remain
    completed=[]
    for run in (ROOT/'runs').glob('*/result.json'):
        if json.loads(run.read_text()).get('completed'):completed.append(run.parent.name)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    record.update(verified=True,stopped_at=time.time(),processes_after=remain,
        preserved_completed=sorted(completed),
        cancelled=[j['id'] for j in manifest['jobs'] if j['id'] not in completed],
        gpu_processes_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True))
    cancellation.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
