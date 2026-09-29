"""Audited controller-only replacement before any job starts; no training restart."""
import json,pathlib,subprocess,time,hashlib
root=pathlib.Path('/home/heechan/optiq-experiments/antmaze-optiq-horizon-temperature-250k-s0-20260925')
old='eee04de7f2c8a34feffda3d0fc376ff9ea1dfe45'
new='7278a0ed0f37e536ae81663fbd1e08a49ae561b8'
state=json.loads((root/'status.json').read_text())
assert not state['running'] and not state['completed'] and not state['failed'] and len(state['pending'])==4
assert not list((root/'jobs').glob('*.json'))
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
status=subprocess.run(ctl+['status',root.name],text=True,capture_output=True)
assert 'STOPPED' in status.stdout,status.stdout
source=pathlib.Path('/home/heechan/OptiQ-ops/sources')/new
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==new
manifest_path=root/'manifest.json';before=manifest_path.read_bytes()
assert json.loads(before)['source_commit']==old
conf=pathlib.Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(root.name+'.conf')
previous=conf.read_text();replacement=previous.replace('directory=/home/heechan/OptiQ-ops/sources/'+old,'directory='+str(source))
assert replacement!=previous and old not in replacement
(root/'controller-conf-before-reservation-fix.ini').write_text(previous)
(root/'status.json').rename(root/'controller-status-before-reservation-fix.json')
conf.write_text(replacement)
provenance=dict(time=time.time(),training_source=old,controller_source=new,
  training_jobs_started_before_replacement=0,training_source_unchanged=True,
  manifest_sha256=hashlib.sha256(before).hexdigest(),
  reason='Reserve predecessor assigned GPU slots during the brief gap before OS flock; no algorithm, settings, or existing learners changed')
(root/'controller-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
subprocess.run(ctl+['reread'],check=True)
subprocess.run(ctl+['update',root.name],check=True)
subprocess.run(ctl+['start',root.name],check=True)
assert manifest_path.read_bytes()==before
print(json.dumps(provenance))
