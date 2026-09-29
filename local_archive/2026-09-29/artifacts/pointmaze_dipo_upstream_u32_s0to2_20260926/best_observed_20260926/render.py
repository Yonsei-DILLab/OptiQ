from pathlib import Path
import json,hashlib,subprocess,sys,concurrent.futures
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parent
REPO=Path('/Users/yunheechan/Documents/ChatGPT/OptiQ/tmp/pointmaze-multiseed-worktree')
sys.path.insert(0,str(REPO))
from maze_benchmarks.visualize_pointmaze import plot_map,plot_rollouts
selection=json.loads((BASE/'selection.json').read_text());chosen=selection['selected']

def collect(r):
 files=['config.json',f"evaluations/{r['step']:09d}_summary.json",f"evaluations/{r['step']:09d}_policy.npz"]
 script=f"import pathlib,json,hashlib\nr=pathlib.Path({r['root']!r})\nprint(json.dumps({{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in {files!r}}}))\n"
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',r['host'],'python3 -'],input=script,text=True,capture_output=True,check=True)
 hashes=json.loads(result.stdout)
 folder=BASE/r['task']
 for relative in files:
  p=folder/relative;p.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['scp','-q','-o','BatchMode=yes',f"{r['host']}:{r['root']}/{relative}",str(p)],capture_output=True,check=True)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[relative]
 summary=json.loads((folder/files[1]).read_text());config=json.loads((folder/'config.json').read_text())
 assert summary['source_commit']==config['source_commit']==r['source']
 with np.load(folder/files[2]) as d:
  ids=d['goal_ids'];counts=np.bincount(ids[ids>=0].astype(int),minlength=len(r['policy']['goals']))
  assert counts.tolist()==r['policy']['goals'];assert len(ids)==r['policy']['episodes'];assert np.isclose(np.mean(ids>=0),r['policy']['success'])
 return {'task':r['task'],'remote_sha256':hashes}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:checks=list(pool.map(collect,chosen))
fig,axes=plt.subplots(1,3,figsize=(15,5.9))
for ax,r in zip(axes,chosen):
 maze=r['task'].removeprefix('pm_');e=r['policy'];folder=BASE/r['task']
 plot_map(ax,maze,goal_counts=e['goals'],outcome_colors=True)
 plot_rollouts(ax,folder/'evaluations'/f"{r['step']:09d}_policy.npz",max_trajectories=e['episodes'],alpha=.45,outcome_colors=True,success_color='#c51b8a',failure_color='#e87924')
 ax.set_title(f"{maze.title()} | DIPO seed {r['seed']}\n{r['step']:,} steps | success {int(sum(e['goals']))}/{e['episodes']} ({e['success']:.0%})",fontsize=13)
 ax.set_xlabel(f"Goals reached: {e['reachable_goals']}/{len(e['goals'])}\nGoal counts: {e['goals']}",fontsize=11)
 ax.set_ylabel('');ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Corrected DIPO | best observed success per maze (post-hoc selection)',fontsize=15,y=.98)
fig.text(.5,.025,'All recorded rollouts shown; overlapping tracks appear as one. Fresh initial Gaussian; reverse diffusion noise OFF.',ha='center',fontsize=10)
fig.subplots_adjust(left=.025,right=.985,bottom=.16,top=.84,wspace=.15)
for ext in ['png','pdf']:fig.savefig(BASE/f'dipo_best_simple_medium_hard.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
selection.update(verification=checks,report_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),all_recorded_episodes_plotted=True)
(BASE/'manifest.json').write_text(json.dumps(selection,indent=2))
print(BASE/'dipo_best_simple_medium_hard.png')
