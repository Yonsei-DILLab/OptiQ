"""Copy evaluation evidence and original checkpoint/config inputs without mutation."""
from pathlib import Path
import subprocess,tarfile,argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
r=p.parse_args().root;r.mkdir(parents=True,exist_ok=True)
remote='''from pathlib import Path
import json,tarfile,sys
r=Path("/home/heechan/optiq-experiments/trg-inference-resampling-20260921")
m=json.loads((r/"manifest.json").read_text())
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz") as t:
 for name in ["manifest.json","runtime.json","results"]:
  if (r/name).exists():t.add(r/name,arcname=name)
 for p in r.glob("*.log"):t.add(p,arcname="logs/"+p.name)
 for j in m["jobs"]:
  for name in ["config","actor","critic"]:
   p=Path(j[name]);t.add(p,arcname="checkpoints/"+j["name"]+"/"+p.name)
'''
with (r/'snapshot.tar.gz').open('wb') as f:
 subprocess.run(['ssh','vast-heechan-199','python3 -'],input=remote.encode(),stdout=f,check=True)
with tarfile.open(r/'snapshot.tar.gz') as t:t.extractall(r,filter='data')
print('Saved manifest, runtime, results, logs, original configs and final actor/critic checkpoints.')
