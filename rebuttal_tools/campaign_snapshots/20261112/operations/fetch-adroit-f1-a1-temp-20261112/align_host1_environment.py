from pathlib import Path
import json,subprocess,shlex,tarfile,hashlib
P=Path(__file__).resolve().parent;root='/workspace/heechan-unified-f1-a1-20261112';a=json.loads((P/'environment-host1.json').read_text());b=json.loads((P/'environment-host4.json').read_text())
changed=[k for k in b['files'] if a['files'].get(k)!=b['files'][k]];extra=[k for k in a['files'] if k not in b['files']]
cmd='tar -cz -C '+shlex.quote(b['module'])+' '+shlex.join(changed)
with (P/'robotics-host4-delta.tgz').open('wb') as f:
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-4',cmd],stdout=f,stderr=subprocess.PIPE,timeout=40);assert not r.returncode,r.stderr
r=subprocess.run(['scp','-q',str(P/'robotics-host4-delta.tgz'),'ibolt-server-1:'+root+'/robotics-host4-delta.tgz'],capture_output=True,timeout=60);assert not r.returncode,r.stderr
code='ROOT='+repr(root)+'\nSOURCE='+repr(a['module'])+'\nEXPECTED='+repr(b['files'])+'\nEXTRA='+repr(extra)+'\n'+r'''
from pathlib import Path
import subprocess,json,shutil,tarfile,hashlib,os
root=Path(ROOT);basepy='/workspace/heechan-adroit/venv/bin/python';venv=root/'runtime';assert not venv.exists();subprocess.run([basepy,'-m','venv','--without-pip',str(venv)],check=True)
site=next((venv/'lib').glob('python*/site-packages'));base=Path(SOURCE).parent
(site/'base_dependencies.pth').write_text(str(base)+'\n');dest=site/'gymnasium_robotics';shutil.copytree(SOURCE,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for n in EXTRA:(dest/n).unlink()
with tarfile.open(root/'robotics-host4-delta.tgz') as t:t.extractall(dest,filter='data')
manifest={str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(dest.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.pyo']};assert manifest==EXPECTED
(root/'robotics-source-manifest.json').write_text(json.dumps(manifest,indent=2));(root/'runtime-provenance.json').write_text(json.dumps(dict(base_dependencies=str(base),robotics_reference_host=4,robotics_reference_module='/workspace/heechan-fetch-sigma/venv/lib/python3.11/site-packages/gymnasium_robotics',all_package_files_sha256_equal=True),indent=2))
p=root/'plan.json';plan=json.loads(p.read_text());plan['python']={'fetch':str(venv/'bin/python'),'adroit':str(venv/'bin/python')};p.write_text(json.dumps(plan,indent=2))
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false')
r=subprocess.run([str(venv/'bin/python'),str(root/'preflight.py')],env=env,text=True,capture_output=True,timeout=180);(root/'preflight-aligned.log').write_text(r.stdout+r.stderr);assert not r.returncode,r.stderr;print(r.stdout)
print(json.dumps(dict(aligned=True,files=len(manifest),python=str(venv/'bin/python'))))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-1','python3 -'],input=code,text=True,capture_output=True,timeout=240);assert not r.returncode,r.stdout+r.stderr;print(r.stdout)
