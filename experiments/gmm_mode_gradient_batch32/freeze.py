"""Create a complete source manifest from a committed tree, not working files."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',default='HEAD');a=p.parse_args()
repo=Path(__file__).resolve().parents[2]
def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
sha=git('rev-parse',a.commit).decode().strip();prefix='analysis_tools/studies/20260917_nonstationary_q/'
parent=json.loads(git('show',sha+':'+prefix+'SOURCE_MANIFEST.json'))
files=[prefix+f for f in parent['files']]+[prefix+'SOURCE_MANIFEST.json',prefix+'PROTOCOL.md','experiments/__init__.py']
files+=git('ls-tree','-r','--name-only',sha,'experiments/gmm_gradient_interference').decode().splitlines()
files+=git('ls-tree','-r','--name-only',sha,'experiments/gmm_mode_gradient').decode().splitlines()
files+=git('ls-tree','-r','--name-only',sha,'experiments/gmm_mode_gradient_batch32').decode().splitlines()
blobs={f:git('show',sha+':'+f) for f in files};hashes={f:hashlib.sha256(b).hexdigest() for f,b in blobs.items()}
assert all(hashes[prefix+f]==h for f,h in parent['files'].items())
m=dict(commit=sha,source_code_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),files=hashes)
a.output.mkdir(parents=True,exist_ok=True);raw=(json.dumps(m,indent=2)+'\n').encode()
(a.output/'SOURCE_MANIFEST.json').write_bytes(raw)
with tarfile.open(a.output/f'source_{sha[:12]}.tar.gz','w:gz') as tar:
    for f,b in {**blobs,'SOURCE_MANIFEST.json':raw}.items():
        t=tarfile.TarInfo(f);t.size=len(b);tar.addfile(t,io.BytesIO(b))
for f in ['PROTOCOL.md','plan.json']:(a.output/f).write_bytes(blobs['experiments/gmm_mode_gradient_batch32/'+f])
print(sha,len(blobs))
