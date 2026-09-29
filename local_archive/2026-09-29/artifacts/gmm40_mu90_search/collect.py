"""Read-only snapshot of committed GMM40 screens and matching mu-only samples."""
from pathlib import Path
import subprocess,tarfile,json,argparse,re
p=argparse.ArgumentParser();p.add_argument('campaign');a=p.parse_args();assert re.fullmatch(r'gmm40-mu90-[a-z0-9-]+',a.campaign)
r=Path(__file__).resolve().parent;remote='''import tarfile,sys,json
from pathlib import Path
r=Path("/home/heechan/optiq-experiments")/CAMPAIGN
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz") as t:
 for n in ["manifest.json","registration.json","preflight.json"]:
  if (r/n).exists():t.add(r/n,arcname=n)
 for d in sorted((r/"results").iterdir()):
  if not d.is_dir():continue
  if d.name=="target":t.add(d,arcname="results/target");continue
  latest=d/"latest.json"
  if not latest.exists():continue
  step=json.loads(latest.read_text())["step"]
  for n in ["config.json","latest.json","model_sizes.json","update_count_audit.json","wandb_status.json",f"checkpoints/step_{step:07d}.bin"]:
   z=d/n
   if z.exists():t.add(z,arcname=str(z.relative_to(r)))
  for z in d.glob("evaluations/*/metrics_mu_only.json"):t.add(z,arcname=str(z.relative_to(r)))
  for z in (d/"evaluations"/f"step_{step:07d}").iterdir():
   if z.is_file() and z.name!="metrics_mu_only.json":t.add(z,arcname=str(z.relative_to(r)))
'''.replace('CAMPAIGN',repr(a.campaign))
archive=r/(a.campaign+'.tar.gz');out=r/a.campaign
with archive.open('wb') as f:subprocess.run(['ssh','vast-heechan-180','python3 -'],input=remote.encode(),stdout=f,check=True)
with tarfile.open(archive) as t:t.extractall(out,filter='data')
rows=[]
for f in sorted((out/'results').glob('*/latest.json')):
 d=json.loads(f.read_text());m=json.loads((f.parent/'evaluations'/('step_%07d'%d['step'])/'metrics_mu_only.json').read_text());rows.append(dict(name=f.parent.name,step=d['step'],near=m['high_density_fraction'],coverage=m['mode_coverage'],mmd2=m['mmd2']))
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
