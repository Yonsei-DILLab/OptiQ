import hashlib,json
from pathlib import Path
import numpy as np
from flax import serialization
from .config import write_json

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(p)

def save_array(path,array):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.save(f,array)
    tmp.replace(p)

def checkpoint(path,data):
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    tmp.write_bytes(serialization.msgpack_serialize(data));tmp.replace(p)

def verify_source(root):
    root=Path(root);m=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    bad=[p for p,h in m['files'].items() if sha(root/p)!=h]
    assert not bad,bad
    return m['code_id']

def complete(out,record):
    out=Path(out)
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()
        and not p.is_symlink() and not p.name.endswith('.tmp') and p.name not in ('ARTIFACTS_SHA256.json','COMPLETE.json','RUNNING.json','FAILED.json')}
    write_json(out/'ARTIFACTS_SHA256.json',files)
    write_json(out/'COMPLETE.json',record)
