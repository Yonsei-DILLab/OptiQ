"""Copy only completed reevaluation folders; never operate training services."""
import json
from pathlib import Path
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
REMOTE='/home/heechan/optiq-experiments/antmaze-dense-random-reevaluation-20260924'
SCRIPT='''from pathlib import Path
import json
root=Path(REMOTE)
out={}
for p in sorted(root.iterdir()):
 if not p.is_dir() or p.name=='logs' or 'smoke' in p.name:continue
 row={}
 for f in ['result.json','progress.json','verification.json']:
  if (p/f).exists():row[f]=json.loads((p/f).read_text())
 log=root/'logs'/(p.name+'.log')
 if log.exists() and 'Traceback' in log.read_text():row['error']=log.read_text()[-3000:]
 out[p.name]=row
print(json.dumps(out))
'''


def one(host):
    code='REMOTE='+repr(REMOTE)+'\n'+SCRIPT
    text=subprocess.check_output(['ssh',host,'python3 -c '+shlex.quote(code)],text=True)
    data=json.loads(text)
    (ROOT/f'status-{host}.json').write_text(json.dumps(data,indent=2)+'\n')
    for name,row in data.items():
        if row.get('result.json',{}).get('completed'):
            subprocess.run(['rsync','-az',f'{host}:{REMOTE}/{name}/',str(ROOT/name)+'/'],check=True)
    print(host,json.dumps(data))
    return data


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        data=list(pool.map(one,['vast-heechan-180','vast-heechan-199']))
    completed=sum(bool(r.get('result.json',{}).get('completed')) for d in data for r in d.values())
    print('completed',completed,'/ 5')
