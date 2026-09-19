from pathlib import Path
import json,sys,os,hashlib
ROOT=Path(__file__).resolve().parent
SPEC=json.loads((ROOT/'BASE_SOURCE.json').read_text());BASE=Path(os.environ.get('OPTIQ_BASE',SPEC['path']))
for p in reversed([str(ROOT),str(BASE),str(BASE/'v5'),str(BASE/'vendor')]):sys.path.insert(0,p)
def verify():
 from nsq.io import verify_source
 assert verify_source(BASE)==SPEC['code_id']
 assert json.loads((BASE/'DEPLOYMENT.json').read_text())['commit']==SPEC['commit']
 return verify_source(ROOT)
