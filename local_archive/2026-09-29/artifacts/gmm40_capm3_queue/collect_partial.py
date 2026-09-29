"""Read-only, immutable evaluation snapshot for an interim GMM40 report."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import io,json,sys,tarfile,time
from pathlib import Path
r=Path('/home/heechan/optiq-experiments/gmm40-trg-capm3-mean1-100k-4seed-20260921')
manifest=json.loads((r/'manifest.json').read_text())
snapshot={'captured_at':time.time(),'manifest':manifest,'status':json.loads((r/'status.json').read_text()),'runs':[]}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    tar.add(r/'results/target/definition.json',arcname='target.json')
    for job in manifest['jobs']:
        folder=r/'results'/job['name']
        if not (folder/'latest.json').exists():continue
        latest=json.loads((folder/'latest.json').read_text())
        cutoff=latest['step']
        history=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
        history=[row for row in history if row['step']<=cutoff]
        item=dict(job=job,latest=latest,history=history,
            config=json.loads((folder/'config.json').read_text()),
            queue=json.loads((r/'jobs'/(job['name']+'.json')).read_text()))
        snapshot['runs'].append(item)
        for step in sorted(set(row['step'] for row in history)):
            if job['seed']!=0 and step not in (100000,cutoff):continue
            ev=folder/'evaluations'/f'step_{step:07d}'
            for name in ('samples.npy','samples_mu_only.npy','metrics.json','metrics_mu_only.json'):
                p=ev/name
                if p.exists():tar.add(p,arcname=f"samples/{job['name']}/{step}/{name}")
    raw=json.dumps(snapshot,indent=2).encode()
    info=tarfile.TarInfo('snapshot.json');info.size=len(raw)
    tar.addfile(info,io.BytesIO(raw))
'''

dest=HERE/'interim'/time.strftime('%Y%m%d-%H%M%S')
dest.mkdir(parents=True)
archive=dest/'snapshot.tar.gz'
with archive.open('wb') as f:
    result=subprocess.run(['ssh','-o','ControlMaster=auto','-o','ControlPersist=600',
        '-o','ControlPath=/tmp/optiq-5090-%r-%h-%p','vast-heechan-180','python3 -'],
        input=REMOTE.encode(),stdout=f,stderr=subprocess.PIPE,timeout=120)
if result.returncode:raise RuntimeError(result.stderr.decode())
with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
data=json.loads((dest/'snapshot.json').read_text())
print(dest)
for run in data['runs']:
    x=run['latest']
    print(run['job']['name'],x['step'],{k:x[k] for k in ['mode_coverage','high_density_fraction','mmd2','mode_mass_tv']})
