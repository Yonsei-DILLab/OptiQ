"""Read-only collection for the nine authorized temperature-only runs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CAMPAIGN = 'antmaze-optiq-dense-dacer-off-T3-T5-T10-s0-20260924'
SOURCE = '5baa5b3463416cdfdcad465e8f862f3729802568'
REMOTE = r'''
import json,subprocess,time
from pathlib import Path
root=Path('/home/heechan/optiq-experiments/antmaze-optiq-dense-dacer-off-T3-T5-T10-s0-20260924')
data={'collected_unix':time.time()}
paths=[root/n for n in ('manifest.json','status.json','registration.json','failure.json','result.json','wandb-sync-status.json')]
paths+=list((root/'jobs').glob('*.json'))
for phase in ('preflight','runs'):
 for run in (root/phase).glob('*'):
  paths += [run/n for n in ('config.json','progress.json','result.json','checkpoint-verification.json','optiq-profile-verification.json','dacer-disabled-verification.json','parameter-audit.json','wandb.json','failure.json')]
for p in paths:
 if p.is_file():data[str(p.relative_to(root))]=json.loads(p.read_text())
for p in (root/'runs').glob('*/evaluations/*/*/summary.json'):
 data[str(p.relative_to(root))]=json.loads(p.read_text())
data['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True)
gpu_pids={int(line.split(',')[0]) for line in data['gpu_processes'].splitlines() if line.strip()}
data['gpu_status']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
data['processes']=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'antmaze_experiments.run ' in cmd and int(p.name) not in gpu_pids:continue
  if str(root) in cmd and ('antmaze_experiments.' in cmd or '/launch.sh ' in cmd):
   data['processes'].append({'pid':int(p.name),'cmd':cmd,'cwd':str((p/'cwd').resolve())})
 except (OSError,UnicodeError):pass
source=Path('/home/heechan/OptiQ-ops/sources/5baa5b3463416cdfdcad465e8f862f3729802568')
data['frozen_head']=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
data['frozen_status']=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True)
data['branch_head']=subprocess.check_output(['git','-C','/home/heechan/OptiQ-direct-gmm-trg-antmaze','rev-parse','HEAD'],text=True).strip()
data['services']={}
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf','status']
for name in (root.name,root.name+'-wandb-sync'):
 data['services'][name]=subprocess.run(ctl+[name],text=True,capture_output=True).stdout.strip()
for name in ('controller.log','controller.err','sync_wandb.err'):
 p=root/name
 if p.exists():data[name]='\n'.join(p.read_text(errors='replace').splitlines()[-10:])
print(json.dumps(data))
'''


def collect(host):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                             'python3', '-'], input=REMOTE, text=True, capture_output=True,
                            check=True, timeout=60)
    data = json.loads(result.stdout)
    (ROOT / (host + '.json')).write_text(json.dumps(data, indent=2) + '\n')
    return host, data


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        for host, data in pool.map(collect, ('vast-heechan-180', 'vast-heechan-199')):
            status = data['status.json']
            print(host, 'pending=', status['pending'], 'failed=', status['failed'])
            for entry in status['running']:
                identifier = entry['id']
                job = data.get('jobs/' + identifier + '.json', {})
                progress = data.get('runs/' + identifier + '/progress.json', {})
                preflight = data.get('preflight/' + identifier + '/result.json', {})
                print(identifier, 'GPU', entry['gpu'], job.get('phase'),
                      'step=', progress.get('step'), 'preflight=', preflight.get('completed', False))
    (ROOT / 'collected-at.json').write_text(json.dumps(dict(
        time=datetime.now(timezone.utc).isoformat(), campaign=CAMPAIGN, source=SOURCE), indent=2) + '\n')


if __name__ == '__main__':
    main()
