"""Freeze numerical source only; reports and operational status may be updated."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
paths=[*root.joinpath('v5').rglob('*'),*root.joinpath('experiment').glob('*.py'),root/'tasks.json']
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
       if p.is_file() and not {'__pycache__','.pytest_cache','.git'}.intersection(p.parts)}
record={'files':files,'code_id':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
(root/'SOURCE_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['code_id'],len(files))
