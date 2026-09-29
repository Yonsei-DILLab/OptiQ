"""Read-only snapshot of corrected DIPO PointMaze trajectories; no training changes."""
import concurrent.futures,datetime,hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent / 'trajectory_latest_20260926T0900Z'
REPO=Path('/Users/yunheechan/Documents/ChatGPT/OptiQ/tmp/pointmaze-multiseed-worktree')
sys.path.insert(0,str(REPO))
CAMPAIGN='pointmaze-dipo-upstream-u32-s0to2-1m-20260926'
OTHER='maze-dipo-upstream-u32-s0-1m-20260926'
JOBS=[('vast-heechan-6',OTHER,'simple',0)]+[('vast-heechan-199' if s==2 else 'vast-heechan-46',CAMPAIGN,m,s) for m in ['medium','hard'] for s in range(3)]
BASE.mkdir(parents=True,exist_ok=True)

def collect(host):
 jobs=[j for j in JOBS if j[0]==host]
 roots=[f'/home/heechan/optiq-experiments/{c}/runs/pm_{m}-dipo-upstream-u32-s{s}' for _,c,m,s in jobs]
 script="""import pathlib,json,hashlib,io,tarfile,sys,time
roots=ROOTS
files={};manifest={'captured_at':time.time(),'files':{}}
for root in roots:
 r=pathlib.Path(root)
 selected=[r/'config.json',r/'progress.json']
 for s in sorted((r/'evaluations').glob('*_summary.json')):
  summary=json.loads(s.read_text());p=r/'evaluations'/f"{summary['step']:09d}_policy.npz"
  if p.exists():selected.extend([s,p])
 for f in selected:
  relative=r.name+'/'+str(f.relative_to(r));data=f.read_bytes()
  files[relative]=data;manifest['files'][relative]={'sha256':hashlib.sha256(data).hexdigest(),'remote':str(f),'bytes':len(data)}
files['remote-manifest.json']=json.dumps(manifest,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for relative,data in files.items():
  info=tarfile.TarInfo(relative);info.size=len(data);tar.addfile(info,io.BytesIO(data))
""".replace('ROOTS',repr(roots))
 proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -'],input=script.encode(),capture_output=True,check=True)
 archive=BASE/(host+'.tar.gz');archive.write_bytes(proc.stdout)
 with tarfile.open(fileobj=io.BytesIO(proc.stdout),mode='r:gz') as tar:
  manifest=json.load(tar.extractfile('remote-manifest.json'))
  for rel,meta in manifest['files'].items():
   assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
   data=tar.extractfile(rel).read();assert hashlib.sha256(data).hexdigest()==meta['sha256']
   dest=BASE/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 (BASE/(host+'-remote-manifest.json')).write_text(json.dumps(manifest,indent=2))
 return {'host':host,'sha256':hashlib.sha256(proc.stdout).hexdigest(),'archive':archive.name,**manifest}
if '--render-only' in sys.argv:
 manifests=json.loads((BASE/'collection-manifest.json').read_text())
else:
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:manifests=list(pool.map(collect,sorted(set(j[0] for j in JOBS))))
 (BASE/'collection-manifest.json').write_text(json.dumps(manifests,indent=2))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from maze_benchmarks.visualize_pointmaze import plot_map,plot_rollouts
records=[]
for host,campaign,maze,seed in JOBS:
 name=f'pm_{maze}-dipo-upstream-u32-s{seed}';root=BASE/name
 cfg=json.loads((root/'config.json').read_text());assert cfg['agent']['diffusion_memory_replace_is_upstream']
 assert cfg['num_envs']==2048 and cfg['updates_per_collect']==32 and cfg['batch_size']==4096
 history=[]
 for f in sorted((root/'evaluations').glob('*_summary.json')):
  summary=json.loads(f.read_text());step=summary['step'];raw=root/'evaluations'/f'{step:09d}_policy.npz'
  with np.load(raw) as d:
   gids=d['goal_ids'];n=len(gids);counts=np.bincount(gids[gids>=0].astype(int),minlength=len(cfg['geometry']['goal_positions']))
   assert counts.tolist()==summary['policy']['goals'];assert np.isclose(np.mean(gids>=0),summary['policy']['success'])
   assert n==summary['policy']['episodes'];assert np.isfinite(d['returns']).all()
   xy=d['xy'];end=np.array([track[np.isfinite(track).all(-1)][-1] for track in xy]);starts=xy[:,0]
   info={'step':step,'updates':summary['updates'],'episodes':n,'success':float(np.mean(gids>=0)),'counts':counts.tolist(),'mean_endpoint':end.mean(0).tolist(),'endpoint_std':end.std(0).tolist(),'start_std':starts.std(0).tolist(),'mean_length':summary['policy']['mean_length']}
   history.append(info)
  assert summary['source_commit']==cfg['source_commit']
 records.append(dict(host=host,name=name,maze=maze,seed=seed,source_commit=cfg['source_commit'],history=history,latest=history[-1]))

def panel(ax,r,e=None):
 e=e or r['latest'];plot_map(ax,r['maze'],goal_counts=e['counts'],outcome_colors=True)
 data=BASE/r['name']/'evaluations'/f"{e['step']:09d}_policy.npz"
 plot_rollouts(ax,data,max_trajectories=e['episodes'],alpha=.5,outcome_colors=True,success_color='#c51b8a',failure_color='#e87924')
 ax.set_title(f"{r['maze'].title()} | seed {r['seed']} | {e['step']:,} steps\nSuccess {e['success']:.0%} ({int(sum(e['counts']))}/{e['episodes']}) | Goals {sum(c>0 for c in e['counts'])}/{len(e['counts'])}",fontsize=12)
 ax.set_xlabel('Goal counts: '+str(e['counts']),fontsize=9);ax.set_ylabel('');ax.set_xticks([]);ax.set_yticks([])

def savefig(selected,rows,cols,name,title):
 fig,axes=plt.subplots(rows,cols,figsize=(4.8*cols,4.6*rows+.9),squeeze=False)
 for ax,(r,e) in zip(axes.flat,selected):panel(ax,r,e)
 for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
 fig.suptitle(title+'\nFresh initial Gaussian per action; reverse diffusion noise OFF (upstream eval)',fontsize=12,y=.985)
 fig.legend(handles=[Line2D([0],[0],color='#c51b8a',lw=3,label='Success'),Line2D([0],[0],color='#e87924',lw=3,label='Failure')],loc='lower center',ncol=2,frameon=False)
 fig.subplots_adjust(left=.03,right=.98,bottom=.085,top=.86 if rows==1 else .91,wspace=.16,hspace=.28)
 fig.savefig(BASE/name,dpi=180,bbox_inches='tight');plt.close(fig)

savefig([(r,None) for r in records if r['maze']!='simple'],2,3,'dipo_latest_medium_hard_seeds012.png','Corrected DIPO | latest completed evaluations at collection time')
savefig([(r,None) for r in records if r['seed']==0],1,3,'dipo_latest_seed0_simple_medium_hard.png','Corrected DIPO seed 0 | latest completed evaluations (different steps)')
simple=next(r for r in records if r['maze']=='simple');hard1=next(r for r in records if r['maze']=='hard' and r['seed']==1)
savefig([(simple,e) for e in simple['history'][-3:]]+[(hard1,e) for e in hard1['history'][-3:]],2,3,'dipo_trajectory_changes_simple_hard1.png','Checkpoint changes | same seed shown separately; no trajectories mixed')
report={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'all_rollouts_plotted':True,'report_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'evaluation':'initial Gaussian ON; reverse noise OFF'}
(BASE/'results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print(BASE)
