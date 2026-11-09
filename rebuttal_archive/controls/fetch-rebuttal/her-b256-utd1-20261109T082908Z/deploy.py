
import pathlib,subprocess,sys,shutil,json,hashlib,os,collections
host=int(sys.argv[1]);payload=pathlib.Path(__file__).resolve().parent
root=pathlib.Path('/workspace/iBOLT');campaign=payload.parent
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()=='heechan-rebuttal'
folder=campaign/f'host{host}';folder.mkdir(exist_ok=False)
source=folder/'source';old=root/f'outputs/fetch-rebuttal/obsnorm-b2048-utd1-20261109T063017Z/host{host}/source'
shutil.copytree(old,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
allowed=[]
for f in (payload/'fetch_rebuttal').rglob('*'):
 if f.is_file():
  rel=f.relative_to(payload);target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);allowed.append(str(rel))
changed=[]
for f in source.rglob('*'):
 if f.is_file():
  rel=f.relative_to(source);prior=old/rel
  if not prior.exists() or f.read_bytes()!=prior.read_bytes():changed.append(str(rel))
assert set(changed)<=set(allowed),(changed,allowed)
sys.path.insert(0,str(source))
from fetch_rebuttal.run import guard
from fetch_rebuttal.her_campaign import make_plan
guard()
usage=collections.Counter()
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  if 'python' in (p/'comm').read_text():
   affinity=os.sched_getaffinity(int(p.name))
   if len(affinity)<=8:usage.update(affinity)
 except (OSError,ProcessLookupError):pass
allowed_cpu=os.sched_getaffinity(0);lanes=set()
for cpu in allowed_cpu:
 siblings=set()
 for part in pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
  if '-' in part:
   low,high=map(int,part.split('-'));siblings.update(range(low,high+1))
  else:siblings.add(int(part))
 lanes.add(tuple(sorted(siblings&allowed_cpu)))
lanes=sorted(lanes,key=lambda c:(sum(usage[x] for x in c),c))[:32]
python='/workspace/iBOLT-runs/fetch-20261108-venv/bin/python'
plan=make_plan(folder,source,python,host);plan['cpu_lanes']=[list(x) for x in lanes]
plan['changed_source_files']=changed
plan['root_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
plan['source_hashes']={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
(folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
program='fetch-'+campaign.name+f'-host{host}'
conf=f'''[program:{program}]
command={python} -u -m fetch_rebuttal.her_campaign --plan {folder}/plan.json
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
stdout_logfile={folder}/controller.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
environment=PYTHONPATH="{source}",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"
'''
subprocess.run(['sudo','-n','tee',f'/etc/supervisor/conf.d/{program}.conf'],input=conf,text=True,stdout=subprocess.DEVNULL,check=True)
subprocess.run(['sudo','-n','supervisorctl','reread'],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['sudo','-n','supervisorctl','add',program],check=True,stdout=subprocess.DEVNULL)
print(json.dumps({'prepared':True,'host':host,'program':program,'folder':str(folder),'jobs':len(plan['jobs']),'queued':sum(j['status']=='queued' for j in plan['jobs']),'changes':changed}),flush=True)
