"""Transfer dispatch ownership without stopping any active training process."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path
from queue import processes,verify,write

p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True)
p.add_argument('--retry-wandb-startup',nargs='*',default=[]);a=p.parse_args()
package=Path(__file__).parent;manifest=json.loads((package/'SCHEDULER_MANIFEST.json').read_text())
assert all(hashlib.sha256((package/f).read_bytes()).hexdigest()==h for f,h in manifest['files'].items())
plan=json.loads((package/'plan.json').read_text());verify(plan)
old=a.old_root.resolve();new=a.new_root.resolve();assert old!=new
name=plan.get('tmux_prefix','mujoco-v2first')+'-'+manifest['commit'][:7]
assert subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,'New scheduler already active'
status=json.loads((old/'STATUS.json').read_text());assert Path(f"/proc/{status['pid']}/cmdline").exists()
# No process signal: the old scheduler retains its children and their terminal.
before=processes();stop=old/'STOP_NEW_RUNS';stop.write_text('Future dispatch transferred to '+str(new)+'\n');changed=time.time()
# Let any already-entered dispatch settle, including a possible short GPU preflight.
deadline=time.time()+180
while True:
 s=json.loads((old/'STATUS.json').read_text())
 if s['time']>changed:break
 if time.time()>deadline:raise RuntimeError('Old dispatcher did not acknowledge a fresh cycle; left stopped, training unchanged')
 time.sleep(2)
new.mkdir(parents=True,exist_ok=True)
failed=old/'FAILED_ATTEMPTS.json'
failures=json.loads(failed.read_text()) if failed.exists() else {}
approved=[]
for key in a.retry_wandb_startup:
    assert key in ('ant_v3_s1','ant_v3_s2'), 'Only user-requested failed startup runs may be retried'
    rec=failures[key];out=Path(rec['out']);f=out/'FAILED.json';error=json.loads(f.read_text())
    assert error['error']=="RuntimeError('W&B upload was not confirmed at step 0')"
    assert not (out/'resume.zip').exists() and not (out/'progress.json').exists()
    assert not any(r['out']==str(out) for r in processes())
    archive=out/'failure_history'/str(int(changed));archive.mkdir(parents=True)
    f.replace(archive/'FAILED.json');write(archive/'SCHEDULER_ATTEMPT.json',rec)
    approved.append(dict(key=key,prior_failure=error,source_commit=plan['numerical_commits']['v3'],
                         start_step=0,wandb_id_preserved=True,archive=str(archive)))
    del failures[key]
write(new/'FAILED_ATTEMPTS.json',failures)
if approved:write(new/'APPROVED_RETRIES.json',approved)
receipt=dict(time=time.time(),source_commits=plan['numerical_commits'],scheduler_commit=manifest['commit'],previous_root=str(old),previous_scheduler_pid=status['pid'],active_before=before,adopted=processes(),training_signals_sent=False,priority_order=plan['order'])
write(new/'ACTIVATION.json',receipt)
command=['python3',str(package/'queue.py'),'--root',str(new)]
subprocess.run(['tmux','new-session','-d','-s',name,shlex.join(command)+' > '+shlex.quote(str(new/'console.log'))+' 2>&1'],check=True)
write(old/'SUPERSEDED.json',dict(time=time.time(),replacement=str(new),scheduler_commit=manifest['commit'],tmux=name))
print(json.dumps(dict(**receipt,tmux=name),indent=2))
