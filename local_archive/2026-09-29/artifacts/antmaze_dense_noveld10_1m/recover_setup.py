from pathlib import Path
import json,shutil,subprocess,time
root=Path('/home/heechan/optiq-experiments/antmaze-dense-noveld10-1m-s0-20260922')
sha='19cadc1a73dbd55f33d9b488750de870452d40b9'
source=Path('/home/heechan/OptiQ-ops/sources')/sha
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
s=json.loads((root/'status.json').read_text())
assert s['stage']=='preflight' and s['phase']=='failed' and s['source_commit']==sha
assert not list((root/'runs').iterdir()),'Never recover an active production campaign this way'
for j in s['jobs']:
 assert j['status']!='running' and not Path('/proc',str(j.get('pid',0))).exists()
 if j['status']=='failed':
  log=(root/'logs'/f"preflight-{j['id']}.log").read_text()
  assert "ModuleNotFoundError: No module named 'jaxrl5'" in log or "ModuleNotFoundError: No module named 'meow_continuous_action'" in log
status=subprocess.run(ctl+['status',root.name],capture_output=True,text=True).stdout
assert 'EXITED' in status or 'FATAL' in status or 'STOPPED' in status,status
expected={'gmm40-baseline/MFPO':'d8b3977d29d4ef2d315e871337e5826f2eb79eb2','gmm40-baseline/meow':'b786d27aa9b03e4242ee8904ff884b21fe65e2f7'}
for path,pinned in expected.items():
 assert subprocess.check_output(['git','-C',str(source/path),'rev-parse','HEAD'],text=True).strip()==pinned
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
archive=root.with_name(root.name+'-preflight-import-failure')
assert not archive.exists();root.rename(archive);root.mkdir();(root/'logs').mkdir()
shutil.copy2(archive/'registration.json',root/'registration.json')
(root/'setup-recovery.json').write_text(json.dumps(dict(manual_setup_recovery=True,production_started_before_recovery=False,source_commit=sha,source_code_changed=False,reason='Initialize the already-pinned MEOW and MFPO Git submodules',dependency_commits=expected,archived_attempt=str(archive),time=time.time()),indent=2)+'\n')
subprocess.run(ctl+['start',root.name],check=True)
print(json.dumps(dict(recovered_setup=True,archived_attempt=str(archive),source_commit=sha)))
