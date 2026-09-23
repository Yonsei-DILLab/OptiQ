"""Expose existing read-only Python packages after isolated overrides."""
import json
from pathlib import Path
import subprocess
import sys

base,target=sys.argv[1:]
paths=json.loads(subprocess.check_output([base,'-c',
    'import json,site;print(json.dumps(site.getsitepackages()))'],text=True))
site=Path(subprocess.check_output([target,'-c',
    'import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True).strip())
content='\n'.join(p for p in paths if Path(p).is_dir())+'\n'
(site/'antmaze_base_runtime.pth').write_text(content)
print('Isolated runtime inherits base dependencies:',paths)
