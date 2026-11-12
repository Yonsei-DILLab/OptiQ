from pathlib import Path
import json,subprocess,hashlib,tarfile
P=Path(__file__).resolve().parent;s=json.loads((P/'evidence.json').read_text())
runs=[]
for r in s['runs']:
 steps=sorted(int(f['name'][12:-8]) for f in r['checkpoints'] if f['name'].startswith('actor_state_'))
 desired=[100000,200000,300000,400000,500000,600000,700000,800000,1000000,1200000,1500000]
 chosen=[min(steps,key=lambda x:abs(x-t)) for t in desired];assert len(set(chosen))==len(desired)
 runs.append(dict(id=r['job']['id'],seed=r['job']['seed'],M=256 if r['job']['phase']=='nm_pilot' else 64,central_path=r['central_path'],steps=chosen))
assert len({tuple(r['steps']) for r in runs})==1
spec=dict(runs=runs,steps=runs[0]['steps']);(P/'probe-spec.json').write_text(json.dumps(spec,indent=2))
code='SPEC='+repr(spec)+'\n'+r'''
from pathlib import Path
import tarfile,sys,json,hashlib,io
hashes={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for r in SPEC['runs']:
  o=Path(r['central_path']);paths=[o/'mujoco-config.json',o/'config.json']
  for s in r['steps']:
   paths.extend([o/'checkpoints'/f'actor_state_{s}.msgpack',o/'checkpoints'/f'critic_state_{s}.msgpack',o/'checkpoints'/f'normalizer_{s:09d}.json'])
  for f in paths:
   name=r['id']+'/'+str(f.relative_to(o));data=f.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();t.add(f,arcname=name)
 data=json.dumps(hashes,indent=2).encode();info=tarfile.TarInfo('checkpoint-sha256.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
'''
archive=P/'checkpoints.tgz'
with archive.open('wb') as out:
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code.encode(),stdout=out,stderr=subprocess.PIPE,timeout=100)
assert not r.returncode,r.stderr
remote='/workspace/heechan-unified-f1-a1-20261112/diagnostics/slide-nm-causal-20261112'
cmd="python3 -c \"from pathlib import Path;Path('"+remote+"').mkdir(parents=True,exist_ok=True)\""
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3',cmd],capture_output=True,timeout=25);assert not r.returncode,r.stderr
with archive.open('rb') as inp:r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3','tar -xz -C '+remote],stdin=inp,capture_output=True,timeout=100)
assert not r.returncode,r.stderr
with tarfile.open(archive) as t:manifest=json.load(t.extractfile('checkpoint-sha256.json'))
(P/'checkpoint-sha256.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(passed=True,archive_mib=archive.stat().st_size/2**20,checkpoint_groups=len(runs)*len(spec['steps']),steps=spec['steps'],remote=remote)))
