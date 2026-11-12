from pathlib import Path
import json,sys,time
from preflight import ssh,ROOT
P=Path(__file__).resolve().parent;TAG='nm-ablation-20261112'
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-entropy-20261112'))
from deploy import freeze,resume

def remote(h,code,**kw):
 return ssh(h,'\n'.join(f'{k}={v!r}' for k,v in dict(ROOT=ROOT,TAG=TAG,HOST=h,**kw).items())+'\n'+code,timeout=55)
def save(name,d):(P/name).write_text(json.dumps(d,indent=2))
mutation=r'''
from pathlib import Path
import json,hashlib,fcntl,collections,time,shutil
p=Path(ROOT);stage=p/TAG;assert (p/'PAUSE').read_text().startswith(TAG)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((p/'plan.json').read_text());before=json.loads((stage/'plan.before.json').read_text());assert plan==before
jobs=json.loads((stage/'additions.json').read_text());ids={j['id'] for j in plan['jobs']}
assert all(j['id'] not in ids and j['host']==HOST and j['phase']=='nm_pilot' and j['status']=='queued' for j in jobs)
assert all(not Path(j['output']).exists() and not (p/'specs'/(j['id']+'.json')).exists() for j in jobs)
assert PROOF['passed'] and PROOF['source_manifests']==json.loads((p/'nm-transfer-proof.json').read_text())['source_manifests']
assert PROOF['protocol_sha256']==hashlib.sha256((p/'nm-protocol.json').read_bytes()).hexdigest()
(p/'nm-archive-ready.json').write_text(json.dumps(PROOF,indent=2))
for name,digest in PROOF['source_manifests'].items():plan['source_manifests'][name.removesuffix('-source')]=digest
shutil.copy2(p/'controller.py',stage/'controller.py.before');shutil.copy2(stage/'controller.py',p/'controller.py')
if HOST==3:
 assert not any(j['status'] in ['running','queued','waiting_selection','waiting_temperature','waiting_entropy'] for j in plan['jobs'])
 assert all(j['family']=='adroit' for j in jobs)
 # Completed Adroit runs show the full model/checkpoint footprint; N/M does not change parameter shapes.
 sizes=[]
 for j in plan['jobs']:
  if j['family']=='adroit' and j['status']=='completed' and j['phase']!='smoke':
   sizes.append(sum(f.stat().st_size for f in Path(j['output']).rglob('*') if f.is_file()))
 assert sizes and max(sizes)<128*2**20 and shutil.disk_usage(p).free>6*2**30
 (stage/'disk-admission-proof.json').write_text(json.dumps(dict(completed_adroit_runs=len(sizes),max_complete_run_mib=max(sizes)/2**20,free_gib=shutil.disk_usage(p).free/2**30,old_min_free_gib=plan['min_free_gib'],new_min_free_gib=4,per_job_reserve_mib=128,note='Only 12 Adroit admissions; no model-size change; central hash-verified archival remains enabled'),indent=2))
 plan['min_free_gib']=4;plan['disk_reserve_mib_per_job']=128
plan['jobs'].extend(jobs);assert plan['jobs'][:len(before['jobs'])]==before['jobs']
plan['nm_protocol']=str(p/'nm-protocol.json');plan['counts']=dict(collections.Counter(j['status'] for j in plan['jobs']));plan['updated_unix']=time.time()
t=p/'plan.nm.tmp';t.write_text(json.dumps(plan,indent=2));t.replace(p/'plan.json')
print(json.dumps(dict(host=HOST,added=len(jobs),counts=plan['counts'],old_jobs_preserved=True)))
'''
if __name__=='__main__':
 for family in ['fetch','adroit']:assert json.loads((P/('numeric-'+family+'.json')).read_text())['passed']
 held={};committed=False
 try:
  for h in [1,2,3]:
   pre=json.loads((P/f'preflight-host{h}.json').read_text());proof=json.loads((P/f'central-source-receipt-host{h}.json').read_text());assert proof['passed']
   assert json.loads((P/f'staged-host{h}.json').read_text())['passed']
   held[h]=remote(h,freeze,OLD_CONTROLLER=pre['hashes']['controller.py'],OLD_GATEWAY=pre['hashes']['archive_gateway.py']);save(f'before-host{h}.json',held[h]);print('HELD',h,flush=True)
  for h in held:
   r=remote(h,mutation,PROOF=json.loads((P/f'central-source-receipt-host{h}.json').read_text()));save(f'admitted-host{h}.json',r);print('APPENDED',r,flush=True)
  for h in held:
   actual=remote(h,"from pathlib import Path\nimport json\np=Path(ROOT);d=json.loads((p/'plan.json').read_text());print(json.dumps(d))")
   assert actual['jobs'][:len(held[h]['jobs'])]==held[h]['jobs']
   assert actual['jobs'][len(held[h]['jobs']):]==json.loads((P/f'additions-host{h}.json').read_text())
  committed=True;save('committed.json',dict(passed=True,at=time.time(),new_runs=32,existing_learner_pids_preserved=True))
 finally:
  if not committed:
   for h in held:
    remote(h,"from pathlib import Path\nimport shutil\np=Path(ROOT);s=p/TAG\nshutil.copy2(s/'plan.before.json',p/'plan.json')\nif (s/'controller.py.before').exists():shutil.copy2(s/'controller.py.before',p/'controller.py')\nprint('{}')")
  for h in held:
   r=remote(h,resume);save(f'resumed-host{h}.json',r);print('RESUMED',h,flush=True)
