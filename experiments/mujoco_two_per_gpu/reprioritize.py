"""Transfer dispatch ownership without stopping any active training process."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path
from queue import processes,verify,write

p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True);a=p.parse_args()
package=Path(__file__).parent;manifest=json.loads((package/'SCHEDULER_MANIFEST.json').read_text())
assert all(hashlib.sha256((package/f).read_bytes()).hexdigest()==h for f,h in manifest['files'].items())
plan=json.loads((package/'plan.json').read_text());verify(plan)
old=a.old_root.resolve();new=a.new_root.resolve();assert old!=new
name='mujoco-v2first-'+manifest['commit'][:7]
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
if failed.exists():write(new/'FAILED_ATTEMPTS.json',json.loads(failed.read_text()))
receipt=dict(time=time.time(),source_commits=plan['numerical_commits'],scheduler_commit=manifest['commit'],previous_root=str(old),previous_scheduler_pid=status['pid'],active_before=before,adopted=processes(),training_signals_sent=False,priority_order=plan['order'])
write(new/'ACTIVATION.json',receipt)
command=['python3',str(package/'queue.py'),'--root',str(new)]
subprocess.run(['tmux','new-session','-d','-s',name,shlex.join(command)+' > '+shlex.quote(str(new/'console.log'))+' 2>&1'],check=True)
write(old/'SUPERSEDED.json',dict(time=time.time(),replacement=str(new),scheduler_commit=manifest['commit'],tmux=name))
print(json.dumps(dict(**receipt,tmux=name),indent=2))
