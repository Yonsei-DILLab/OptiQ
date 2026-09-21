"""Run only on a fresh deployment copied from the committed source directory."""
from pathlib import Path
import hashlib,json,argparse,time
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
assert len(a.commit)==40 and not (root/'SOURCE_MANIFEST.json').exists()
files={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*'))
 if f.is_file() and '__pycache__' not in f.parts and f.name!='.DS_Store'}
code=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
(root/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(files=files,code_id=code),indent=2)+'\n')
(root/'DEPLOYMENT.json').write_text(json.dumps(dict(commit=a.commit,source_code_id=code,created=time.time(),
 base=json.loads((root/'BASE_SOURCE.json').read_text())),indent=2)+'\n')
print(code,len(files))
