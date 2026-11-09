"""Prepare isolated sources and supervised preflights; does not stop any training."""
import collections, hashlib, json, os, pathlib, shutil, subprocess, sys

name='mfpo-grid-her-b256-20261109T101314Z'
root=pathlib.Path('/workspace/iBOLT')
oldfolder=root/'outputs/fetch-rebuttal/her-b256-utd1-20261109T082908Z/host2'
folder=root/'outputs/fetch-rebuttal'/name/'host2'
payload=pathlib.Path(__file__).resolve().parent
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()=='heechan-rebuttal'
folder.mkdir(parents=True,exist_ok=False)
source=folder/'source';oldsource=oldfolder/'source'
shutil.copytree(oldsource,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
allowed=[]
for f in (payload/'fetch_rebuttal').rglob('*'):
    if f.is_file():
        rel=f.relative_to(payload);shutil.copy2(f,source/rel);allowed.append(str(rel))
changed=[]
for f in source.rglob('*'):
    if f.is_file():
        rel=f.relative_to(source);prior=oldsource/rel
        if not prior.exists() or f.read_bytes()!=prior.read_bytes():changed.append(str(rel))
assert set(changed)<=set(allowed),(changed,allowed)
sys.path.insert(0,str(source))
from fetch_rebuttal.run import guard,save_json
from fetch_rebuttal.mfpo_grid_campaign import make_jobs
guard()
manifest=json.loads((source/'fetch_rebuttal/SOURCE.json').read_text())
for n,digest in manifest['baseline_origins']['vendors']['mfpo']['files'].items():
    assert hashlib.sha256((source/'fetch_rebuttal/baselines/vendor/mfpo'/n).read_bytes()).hexdigest()==digest,n
old=json.loads((oldfolder/'plan.json').read_text())
usage=collections.Counter()
for p in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        if 'python' in (p/'comm').read_text():
            affinity=os.sched_getaffinity(int(p.name))
            if len(affinity)<=8:usage.update(affinity)
    except (OSError,ProcessLookupError):pass
cpus=os.sched_getaffinity(0);lanes=set()
for cpu in cpus:
    siblings=set()
    for part in pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));siblings.update(range(lo,hi+1))
        else:siblings.add(int(part))
    lanes.add(tuple(sorted(siblings&cpus)))
oldlanes=[tuple(x) for x in old['cpu_lanes']]
fresh=sorted(lanes-set(oldlanes),key=lambda c:(sum(usage[x] for x in c),c))[:16]
plan=dict(old)
plan.update(jobs=make_jobs(folder,source),status='prepared',controller_source=str(source),
    source=str(oldsource),campaign=name,mfpo_preflights={},mfpo_selection=None,
    cpu_lanes=[list(c) for c in fresh+oldlanes],changed_source_files=changed,
    source_hashes={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in source.rglob('*') if f.is_file() and '__pycache__' not in f.parts},
    previous_plan=str(oldfolder/'plan.json'),host=2)
plan.pop('controller_pid',None);plan.pop('counts',None)
save_json(folder/'plan.json',plan)
for suffix,extra in [('-preflight',' --preflight-only'),('','')]:
    program='fetch-'+name+'-host2'+suffix
    config=f'''[program:{program}]
command={plan['python']} -u -m fetch_rebuttal.mfpo_grid_campaign --plan {folder}/plan.json{extra}
directory={source}
user=heechan
autostart=false
autorestart=false
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={folder}/controller{suffix}.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
environment=PYTHONPATH="{source}",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"
'''
    subprocess.run(['sudo','-n','tee',f'/etc/supervisor/conf.d/{program}.conf'],input=config,text=True,
                   stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sudo','-n','supervisorctl','reread'],stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sudo','-n','supervisorctl','add',program],stdout=subprocess.DEVNULL,check=True)
program='fetch-'+name+'-host2-preflight'
subprocess.run(['sudo','-n','supervisorctl','start',program],check=True)
print(json.dumps(dict(prepared=True,folder=str(folder),program=program,jobs=len(plan['jobs']),
                      changed=changed,cpu_lanes=len(plan['cpu_lanes']))),flush=True)
