import json,subprocess
from pathlib import Path
repo=Path('/home/heechan/OptiQ-direct-gmm-trg')
source='a2328f45b3604f1ab3f6e2b2117f7f21ca6d0b71'
def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
assert git('branch','--show-current')=='direct-gmm-trg'
assert not git('status','--porcelain')
assert not (repo/'gmm40').exists() and not (repo/'gmm40-baseline').exists()
for folder in ('common','models','diffusion'):
 assert git('rev-parse',source+':'+folder)==git('rev-parse','HEAD:'+folder),folder+' dependency differs'
files=git('ls-tree','-r','--name-only',source,'--','gmm40').splitlines()
for name in files:
 p=repo/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes(subprocess.check_output(['git','show',source+':'+name],cwd=repo))
legacy=repo/'gmm40/_v5';legacy.mkdir()
snapshot=git('ls-tree','-r','--name-only',source,'--','optiq_dime','configs','LICENSE').splitlines()
for name in snapshot:
 p=legacy/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes(subprocess.check_output(['git','show',source+':'+name],cwd=repo))
(legacy/'__init__.py').write_text('"""Pinned v5 implementation for the historical GMM40 adapters."""\n')
(repo/'gmm40/__init__.py').write_text('"""Historical v5 GMM40 experiments, isolated from the active TRG implementation."""\nfrom pathlib import Path\nV5_SOURCE = Path(__file__).resolve().parent / "_v5"\n')
for p in (repo/'gmm40').glob('*.py'):
 s=p.read_text().replace('from optiq_dime', 'from gmm40._v5.optiq_dime')
 if p.name=='navigation_run.py':
  s=s.replace('from .target import ROOT,RESULTS,initialize_target','from .target import ROOT,RESULTS,initialize_target\nfrom . import V5_SOURCE')
  s=s.replace("str(ROOT/'configs')","str(V5_SOURCE/'configs')")
 if p.name in ('audit_teacher_weights.py','audit_v5_adapter.py'):
  s=s.replace('from gmm40.target import', 'from gmm40 import V5_SOURCE\nfrom gmm40.target import')
  s=s.replace("ROOT/'optiq_dime/", "V5_SOURCE/'optiq_dime/").replace('ROOT / "optiq_dime/', 'V5_SOURCE / "optiq_dime/')
 p.write_text(s)
provenance=dict(source_branch='v5-gmm40',source_commit=source,imported_gmm40_files=files,
 legacy_snapshot=snapshot,shared_unchanged_dependencies={p:git('rev-parse',source+':'+p) for p in ('common','models','diffusion')},
 adaptations=['Namespace v5 imports under gmm40._v5.optiq_dime','Use pinned v5 Hydra config for navigation','Point v5 audit paths at pinned snapshot','Pin upstream repositories as Git submodules'],
 note='Historical OT adapter and its sigma settings retained. Does not select Direct GMM/TRG or change active RL defaults.')
(repo/'gmm40/IMPORT_SOURCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
ignore=repo/'.gitignore'
ignore.write_text(ignore.read_text().rstrip()+'\n\n# Generated GMM40 experiment artifacts.\n/gmm40-results/\n')
for baseline in json.loads((repo/'gmm40/baselines.json').read_text()):
 origin=Path('/home/heechan/OptiQ-v5-gmm40/gmm40-baseline')/baseline['name']
 dest=repo/'gmm40-baseline'/baseline['name']
 subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(origin),str(dest)],check=True)
 subprocess.run(['git','-C',str(dest),'remote','set-url','origin',baseline['url']],check=True)
 subprocess.run(['git','-C',str(dest),'checkout','--detach',baseline['commit']],check=True)
 subprocess.run(['git','submodule','add','--force',baseline['url'],str(dest.relative_to(repo))],cwd=repo,check=True)
subprocess.run(['git','submodule','absorbgitdirs'],cwd=repo,check=True)
print(json.dumps(dict(gmm40_files=len(files),legacy_files=len(snapshot),source_commit=source)))
