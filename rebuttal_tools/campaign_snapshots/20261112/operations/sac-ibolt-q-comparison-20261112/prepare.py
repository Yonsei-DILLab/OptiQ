from pathlib import Path
import json,subprocess,math
P=Path(__file__).resolve().parent;inventory=json.loads((P/'inventory.json').read_text())
rows=inventory['rows'];selected=[];missing=[]
def group(r):return 'ibolt' if r['job']['method']=='ibolt' else 'sac_zero' if r['job']['phase']=='sac_alpha0_pilot' else 'sac_auto'
def available(r):
 names={x['name'] for x in r['checkpoints']}
 if r['job']['family']=='adroit':
  return {5000192} if ('final.jax.msgpack' in names if r['job']['method']=='sac' else 'critic_state_5000192.msgpack' in names) else set()
 prefix='critic_state_' if r['job']['method']=='ibolt' else 'policy_';suffix='.msgpack' if r['job']['method']=='ibolt' else '.jax.msgpack'
 return {int(n[len(prefix):-len(suffix)]) for n in names if n.startswith(prefix) and n.endswith(suffix)}
grid=[100096,200192,500224,1000192,2000128,3000064,4000000,5000192]
for family,tasks in [('fetch',['reach','push','pick-and-place','slide']),('adroit',['door','hammer','pen','relocate'])]:
 for task in tasks:
  rr=[r for r in rows if (r['job']['family'],r['job']['task'])==(family,task)]
  base=[r for r in rr if group(r)!='sac_zero'];assert len(base)==4
  common=set.intersection(*(available(r) for r in base));steps=[s for s in grid if s in common] if family=='fetch' else sorted(common)
  assert steps,(family,task)
  zero=[r for r in rr if group(r)=='sac_zero'];assert len(zero)==2
  zero_steps=sorted(set.intersection(common,*(available(r) for r in zero)))
  zero_steps=[s for s in zero_steps if s in grid]
  for r in rr:
   g=group(r);ss=zero_steps if g=='sac_zero' else steps
   if not ss:missing.append(dict(id=r['job']['id'],reason='No archived checkpoint at a common step yet'));continue
   j=r['job'];selected.append(dict(id=j['id'],family=family,task=task,seed=j['seed'],method=g,host=r['host'],root=r['root'],steps=ss,demo_path=j.get('demo_path'),demo_sha256=j.get('demo_sha256')))
(P/'selection.json').write_text(json.dumps(dict(rows=selected,missing=missing,inventory_unix=inventory['at']),indent=2)+'\n')
code='ROWS='+repr(selected)+'\n'+r'''
from pathlib import Path
import tarfile,sys,json,io,hashlib
hashes={};mapping=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for r in ROWS:
  root=Path(r['root']);files=[root/'config.json'];chunks=[]
  if r['method']=='ibolt':files.append(root/('mujoco-config.json' if r['family']=='fetch' else 'ibolt-config.json'))
  for s in r['steps']:
   if r['method']=='ibolt':f=root/'checkpoints'/f'critic_state_{s}.msgpack'
   elif r['family']=='adroit':f=root/'checkpoints/final.jax.msgpack'
   else:f=root/'checkpoints'/f'policy_{s:09d}.jax.msgpack'
   files.append(f);chunk=dict(step=s,file=str(f.relative_to(root)))
   if r['family']=='fetch':
    n=root/'checkpoints'/f'normalizer_{s:09d}.json';files.append(n);chunk['normalizer']=str(n.relative_to(root))
   chunks.append(chunk)
  for f in files:
   name=r['id']+'/'+str(f.relative_to(root));hashes[name]=hashlib.sha256(f.read_bytes()).hexdigest();t.add(f,arcname=name)
  mapping.append(dict(**r,chunks=chunks))
 for name,obj in [('probe-spec.json',dict(rows=mapping)),('checkpoint-sha256.json',hashes)]:
  raw=json.dumps(obj,indent=2).encode();info=tarfile.TarInfo(name);info.size=len(raw);t.addfile(info,io.BytesIO(raw))
'''
with (P/'probe-checkpoints.tgz').open('wb') as f:
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code.encode(),stdout=f,stderr=subprocess.PIPE,timeout=150);assert not r.returncode,r.stderr
remote='/workspace/heechan-unified-f1-a1-20261112/diagnostics/sac-ibolt-q-comparison-20261112'
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3','mkdir -p '+remote],capture_output=True,timeout=25);assert not r.returncode,r.stderr
with (P/'probe-checkpoints.tgz').open('rb') as f:
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3','tar -xz -C '+remote],stdin=f,capture_output=True,timeout=150);assert not r.returncode,r.stderr
print(json.dumps(dict(runs=len(selected),checkpoints=sum(len(r['steps']) for r in selected),missing=missing,archive_mib=round((P/'probe-checkpoints.tgz').stat().st_size/2**20,1))),flush=True)
