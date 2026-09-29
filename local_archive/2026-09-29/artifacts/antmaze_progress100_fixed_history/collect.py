from pathlib import Path
import subprocess,tarfile,hashlib,json
ROOT=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import io,json,hashlib,tarfile,sys
root=Path('/home/heechan/optiq-experiments/antmaze-progress100-fixed-history-20260924')
base=Path('/home/heechan/optiq-experiments/antmaze-optiq-progress100-2x2-T1-s0-20260924/runs')
watch=Path('/home/heechan/optiq-experiments/antmaze-matched-start-eval-20260924-r2/runs')
files=[]
for p in root.glob('*.json'):files.append((p,'audit/'+p.name))
for p in root.glob('runs/*/step_*'):
 if not (p/'verification.json').exists():continue
 assert json.loads((p/'verification.json').read_text())['passed']
 for q in list(p.glob('*.json'))+list(p.glob('evaluations/*/*/*')):
  if q.is_file() and q.suffix in ('.json','.npz'):files.append((q,'audit/'+str(q.relative_to(root))))
for task in ['v3','v4']:
 name=task+'-optiq-progress100_geodesic_no_bonus-T1-s0';run=base/name
 for rel in ['config.json','learner/progress.csv','progress.json']:
  p=run/rel
  if p.exists():files.append((p,'training/'+name+'/'+rel))
 for p in run.glob('evaluations/*/policy-natural/*'):
  if p.suffix in ('.json','.npz'):files.append((p,'training/'+name+'/'+str(p.relative_to(run))))
 completed=[p for p in (watch/name).glob('step_*') if (p/'result.json').exists()]
 for folder in sorted(completed):
  for p in list(folder.glob('*.json'))+list(folder.glob('evaluations/*/*/*')):
   if p.is_file() and p.suffix in ('.json','.npz'):files.append((p,'watch/'+name+'/'+str(p.relative_to(watch/name))))
checks={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for p,name in files:
  raw=p.read_bytes();info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw));checks[name]=hashlib.sha256(raw).hexdigest()
 raw=json.dumps(checks).encode();info=tarfile.TarInfo('sha256.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
'''
file=ROOT/'snapshot.tar.gz'
with file.open('wb') as f:p=subprocess.run(['ssh','vast-heechan-199','python3','-'],input=REMOTE.encode(),stdout=f,stderr=subprocess.PIPE,timeout=120)
assert p.returncode==0,p.stderr.decode()
with tarfile.open(file) as tar:tar.extractall(ROOT/'data',filter='data')
for name,digest in json.loads((ROOT/'data/sha256.json').read_text()).items():assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest
print('Verified',len(json.loads((ROOT/'data/sha256.json').read_text())),'files')
