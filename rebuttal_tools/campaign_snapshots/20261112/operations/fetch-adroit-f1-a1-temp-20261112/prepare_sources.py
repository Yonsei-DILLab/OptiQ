from pathlib import Path
import subprocess,json,hashlib,tarfile,io,shutil,ast
P=Path(__file__).resolve().parent
# Fetch worker is the verified corrected-HER worker; adapt its SAC import to the shared module.
remote="from pathlib import Path;import sys;sys.stdout.write(Path('/workspace/heechan-fetch-slide-sac-jax/source-fetch-slide-sac-jax-5m-20261111T153417Z/rebuttal_tools/fetch_jax_sac_worker.py').read_text())"
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3','python3 -c '+__import__('shlex').quote(remote)],capture_output=True,text=True,timeout=30);assert not r.returncode,r.stderr
s=r.stdout.replace('fetch_rebuttal.baselines.jax_sac','baselines.jax_sac')
s=s.replace("source/'fetch_rebuttal/baselines/jax_sac.py'","source/'baselines/jax_sac.py'")
s=s.replace("assert method=='sac' and spec['depth'] in (2,3)","assert method=='sac' and spec['depth']==2")
s=s.replace("algorithm='Stable-Baselines3 SAC'","algorithm='JAX SAC with original SB3 initialization'")
ast.parse(s);(P/'fetch-source/rebuttal_tools/fetch_jax_sac_worker.py').write_text(s)
for f in ['baselines/jax_sac.py','rebuttal_tools/check_fetch_jax_sac.py']:
 src=P/'adroit-source'/f;dst=P/'fetch-source'/f;dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
f=P/'fetch-source/rebuttal_tools/fetch_suite_5m_worker.py';s=f.read_text();s=s.replace("assert method in ('sac', 'ibolt')","assert method == 'ibolt', 'SAC must use the shared JAX worker'")
s=s.replace("parameter_selection='Best existing completed five-seed last5 mean: 58.4%; fixed before this comparison'","parameter_selection='Fixed five-temperature pilot grid; confirm only on independent seeds'")
s=s.replace("cfg.dacer.noise_scale=2.; cfg.dacer.target_entropy_per_dim=-.9","cfg.dacer.noise_scale=2.; cfg.dacer.target_entropy_per_dim=-.9\n            assert cfg.dacer.enabled is True\n            assert not cfg.alg.critic.get('layer_norm', False)")
f.write_text(s);ast.parse(s)
for name in ['fetch-source','adroit-source']:
 root=P/name
 old=json.loads((root/'source-sha256.json').read_text())
 (P/(name+'-original-manifest.json')).write_text(json.dumps(old,indent=2))
 files={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and f.name!='source-sha256.json' and '.git' not in f.parts}
 (root/'source-sha256.json').write_text(json.dumps(files,indent=2,sort_keys=True))
 print(name,'changed',[f for f in files if files[f]!=old.get(f)])
with tarfile.open(P/'sources.tgz','w:gz') as t:
 for name in ['fetch-source','adroit-source','fetch-reference.json']:t.add(P/name,arcname=name)
# Fixed expert replay datasets, including failed collection history and episode records.
data=json.loads(Path('outputs/adroit-preparation-20261110/adroit-jax-sac-validation-20261112/expert-datasets.json').read_text())
code='DATA='+repr(data)+'\n'+r'''
from pathlib import Path
import tarfile,sys,io
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for task,d in DATA.items():t.add(Path(d['demonstration_manifest']).parent,arcname='datasets/'+task,filter=lambda m:None if '__pycache__' in m.name else m)
sys.stdout.buffer.write(b.getvalue())
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3','python3 -'],input=code.encode(),capture_output=True,timeout=60);assert not r.returncode,r.stderr
(P/'datasets.tgz').write_bytes(r.stdout);(P/'datasets-original.json').write_text(json.dumps(data,indent=2))
print('datasets',len(r.stdout))
