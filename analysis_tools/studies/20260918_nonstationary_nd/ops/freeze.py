from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['SOURCE_MANIFEST.json'] and '__pycache__' not in p.parts and '.DS_Store' not in p.parts}
record={'files':files,'code_id':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
(root/'SOURCE_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['code_id'],len(files))
