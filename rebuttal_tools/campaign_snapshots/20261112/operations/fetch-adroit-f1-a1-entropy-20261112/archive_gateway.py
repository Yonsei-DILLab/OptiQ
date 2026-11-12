#!/usr/bin/python3
"""Restricted central archive access: read allowed experiment trees and hash-ack checkpoint moves."""
from pathlib import Path
import os,json,shlex,subprocess,hashlib,time,re,sys
BASE=Path(__file__).resolve().parent
CONFIG=json.loads((BASE/'archive-access.json').read_text())
WORK=Path('/workspace')

def allowed(rel):
 p=Path(rel)
 assert not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in CONFIG['allowed_roots']
 f=WORK/p
 assert f.resolve()==f and f.stat().st_uid==os.getuid(),rel
 return f

def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def eligible():
 rows=[]
 cachefile=BASE/'archive-hash-cache.json'
 try:cache=json.loads(cachefile.read_text())
 except (FileNotFoundError,json.JSONDecodeError):cache={}
 for folder in CONFIG['checkpoint_roots']:
  root=allowed(folder)
  for c in root.rglob('checkpoints'):
   if not c.is_dir() or c.is_symlink():continue
   files=[p for p in c.iterdir() if p.is_file() and not p.is_symlink()]
   def step(p):
    nums=re.findall(r'(?<![a-zA-Z])\d{5,}',p.name)
    return int(nums[-1]) if nums else None
   steps=sorted({step(p) for p in files if step(p) is not None})
   keep=set(steps[-2:])
   for p in files:
    n=step(p)
    if n is None or n in keep or time.time()-p.stat().st_mtime<120:continue
    if p.stat().st_uid!=os.getuid() or p.resolve()!=p:continue
    st=p.stat();key=str(p);sig=[st.st_size,st.st_mtime_ns,st.st_ino]
    entry=cache.get(key)
    if not entry or entry['sig']!=sig:entry=dict(sig=sig,sha256=hashfile(p));cache[key]=entry
    rows.append(dict(path=str(p.relative_to(WORK)),size=st.st_size,sha256=entry['sha256']))
 tmp=cachefile.with_name(cachefile.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(cache));tmp.replace(cachefile)
 return rows

cmd=os.environ.get('SSH_ORIGINAL_COMMAND','')
args=shlex.split(cmd)
if args and args[0]=='rsync':
 assert len(args)>3 and args[1:3]==['--server','--sender'] and '.' in args
 rel=args[-1].lstrip('./')
 assert '..' not in Path(rel).parts and Path(rel).parts[0] in CONFIG['allowed_roots'],rel
 os.execv('/usr/bin/rrsync',['rrsync','-ro','/workspace'])
elif args==['checkpoint-manifest']:
 print(json.dumps(eligible()))
elif args==['archive-prune']:
 req=json.load(sys.stdin);current={r['path']:r for r in eligible()};done=[]
 for r in req:
  if not (WORK/r['path']).exists():continue
  assert r['path'] in current and r==current[r['path']],r['path']
  p=allowed(r['path']);assert p.parent.name=='checkpoints' and hashfile(p)==r['sha256']
  p.unlink();done.append(r)
 with (BASE/'archive-moves.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),central=CONFIG['central'],files=done))+'\n')
 print(json.dumps(dict(moved=len(done),bytes=sum(r['size'] for r in done))))
elif args==['archive-ready']:
 d=json.load(sys.stdin);assert d['passed'] and d['hashes']==json.loads((BASE/'transfer-proof.json').read_text())
 (BASE/'archive-ready.json').write_text(json.dumps(d,indent=2));print('archive verified')
elif args==['smoke-gates']:
 d=json.load(sys.stdin);assert set(d)<=set(t+'-'+m for t in ['reach','push','pick-and-place','slide'] for m in ['ibolt','sac']) and all(isinstance(v,bool) for v in d.values())
 (BASE/'fetch-smoke-gates.json').write_text(json.dumps(d,indent=2));print('smoke gates saved')
elif args==['selection']:
 d=json.load(sys.stdin)
 assert d['family'] in ['fetch','adroit'] and d['temperature'] in [.05,.1,.25,.5,1.] and d['pilot_seeds']==[5,6] and d['expected_steps']==5000192
 assert d['task'] in (['reach','push','pick-and-place','slide'] if d['family']=='fetch' else ['door','hammer','pen','relocate'])
 p=BASE/'selections';p.mkdir(exist_ok=True);dest=p/(d['family']+'-'+d['task']+'.json')
 if dest.exists():assert json.loads(dest.read_text())==d
 else:dest.write_text(json.dumps(d,indent=2))
 print('selection saved')
elif args==['entropy-selection']:
 d=json.load(sys.stdin)
 assert d['family'] in ['fetch','adroit'] and d['temperature'] in [.05,.1,.25,.5,1.]
 assert d['target_entropy_per_dim'] in [-.9,0.,.25,.5,.75] and d['pilot_seeds']==[5,6] and d['expected_steps']==5000192
 key=d['family']+'-'+d['task'];assert key in json.loads((BASE/'entropy-protocol.json').read_text())['task_keys']
 tp=BASE/'entropy-temperature-selections'/(key+'.json')
 if not tp.exists():tp=BASE/'selections'/(key+'.json')
 ts=json.loads(tp.read_text());assert ts['temperature']==d['temperature']
 p=BASE/'entropy-selections';p.mkdir(exist_ok=True);dest=p/(key+'.json')
 if dest.exists():assert json.loads(dest.read_text())==d
 else:
  tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(dest)
 print('entropy selection saved')
elif args==['entropy-archive-ready']:
 d=json.load(sys.stdin);assert d['passed']
 assert d['source_manifests']==json.loads((BASE/'entropy-transfer-proof.json').read_text())['source_manifests']
 assert d['protocol_sha256']==hashfile(BASE/'entropy-protocol.json')
 tmp=BASE/'entropy-archive-ready.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(BASE/'entropy-archive-ready.json')
 print('entropy archive verified')
else:raise SystemExit('Restricted archive command rejected')
