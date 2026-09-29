from pathlib import Path
import subprocess,tarfile,json
root=Path(__file__).resolve().parent
remote='''import tarfile,sys
from pathlib import Path
r=Path("/home/heechan/optiq-experiments/gmm40-mu90-screen1-20260921")
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz") as t:
 for n in ["manifest.json","registration.json","preflight.json"]:t.add(r/n,arcname=n)
 for d in sorted((r/"results").iterdir()):
  if not d.is_dir():continue
  if d.name=="target":
   t.add(d,arcname="results/target");continue
  for n in ["config.json","latest.json","model_sizes.json","update_count_audit.json","wandb_status.json","checkpoints/step_0100000.bin"]:
   p=d/n
   if p.exists():t.add(p,arcname=str(p.relative_to(r)))
  for p in d.glob("evaluations/*/metrics_mu_only.json"):t.add(p,arcname=str(p.relative_to(r)))
  for p in (d/"evaluations/step_0100000").iterdir():
   if p.is_file() and p.name!="metrics_mu_only.json":t.add(p,arcname=str(p.relative_to(r)))
'''
archive=root/'screen1.tar.gz'
with archive.open('wb') as f:subprocess.run(['ssh','vast-heechan-180','python3 -'],input=remote.encode(),stdout=f,check=True)
with tarfile.open(archive) as t:t.extractall(root/'screen1',filter='data')
rows=[]
for f in sorted((root/'screen1/results').glob('*/evaluations/step_0100000/metrics_mu_only.json')):
 d=json.loads(f.read_text());rows.append(dict(name=f.parents[2].name,near=d['high_density_fraction'],coverage=d['mode_coverage'],mmd2=d['mmd2']))
(root/'screen1-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
