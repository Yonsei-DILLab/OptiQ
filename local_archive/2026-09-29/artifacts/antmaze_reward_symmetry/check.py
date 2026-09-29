from pathlib import Path
import importlib.util,json,heapq,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
root=Path(__file__).resolve().parent
p=Path('tmp/reward-progress-worktree/antmaze_experiments/progress_reward.py');s=importlib.util.spec_from_file_location('progress',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
result={};fig,axs=plt.subplots(1,2,figsize=(11,5.7));fig.subplots_adjust(top=.83,bottom=.20,wspace=.24)
for ax,task in zip(axs,['v3','v4']):
 g=m.geodesic(task);nodes=np.vstack([[0.,0.],g.vertices]);n=len(nodes)
 adj=np.linalg.norm(nodes[:,None,:]-nodes[None,:,:],axis=-1)
 for i in range(n):adj[i,~g.visible(nodes[i],nodes)]=np.inf
 best=np.full(n,np.inf);best[0]=0.;prev=np.full(n,-1);queue=[(0.,0)]
 while queue:
  dist,u=heapq.heappop(queue)
  if dist>best[u]:continue
  for v in range(n):
   cand=dist+adj[u,v]
   if cand<best[v]-1e-12:best[v]=cand;prev[v]=u;heapq.heappush(queue,(cand,v))
 paths=[]
 for j,goal in enumerate(g.goals):
  idx=np.flatnonzero((nodes==goal).all(-1))[0];route=[idx]
  while route[-1]!=0:route.append(int(prev[route[-1]]))
  xy=nodes[route[::-1]];assert np.isclose(best[idx],g.distances([0.,0.])[j],atol=1e-8)
  paths.append(dict(goal=goal.tolist(),geodesic=float(best[idx]),euclidean=float(np.linalg.norm(goal)),path=xy.tolist()))
  ax.plot(xy[:,0],xy[:,1],c=['#2563eb','#d98513'][j],lw=2.2,label=f"Goal {j+1}: {best[idx]:.2f} m")
  ax.scatter(*goal,c=['#2563eb','#d98513'][j],marker='*',s=130,zorder=8)
 walls,_,bounds=m.maze_geometry(task)
 for x0,y0,x1,y1 in walls:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fc='#e3e7ec',ec='#b5bfcb',lw=.5,zorder=0))
 ax.scatter(0,0,c='black',marker='*',s=110,zorder=8);ax.set(xlim=bounds[[0,2]],ylim=bounds[[1,3]],xlabel='x (m)',ylabel='y (m)',aspect='equal',title=f'{task.upper()} | '+('asymmetric walls' if task=='v3' else 'mirror-symmetric walls'));ax.legend(fontsize=9,loc='lower right')
 result[task]=paths
assert abs((math.sqrt(136)+math.sqrt(40))-result['v3'][0]['geodesic'])<1e-5
assert abs(math.sqrt(288)-result['v3'][1]['geodesic'])<1e-5
# Full distance-field reflection check at matched free positions, not only origin.
xy=np.array([[x,y] for x in np.linspace(-17,1,19) for y in np.linspace(-9,9,19)])
g=m.geodesic('v4');mirror=xy*np.array([1.,-1.]);mask=g.valid(xy)&g.valid(mirror);d1=g.distances(xy[mask]);d2=g.distances(mirror[mask]);error=float(np.max(np.abs(d1-d2[:,::-1])));assert error<1e-8
result['v4_reflection_check']=dict(points=int(mask.sum()),max_goal_swapped_distance_error=error)
result['geometry_sha256']=hashlib.sha256(m.UPSTREAM.read_bytes()).hexdigest();result['point_agent_only']=True;result['training_changed']=False
(root/'result.json').write_text(json.dumps(result,indent=2))
fig.suptitle('Original DDiffPG maps: shortest XY paths used by our geodesic reward',fontsize=13,y=.97)
fig.text(.5,.035,'Point-agent geometry; wall margin 1e-6 m. These lines are distance calculations, not Ant rollouts.\nV3 left goal: wall detour. V4 upper/lower: equal geometry and reward under reflection.',ha='center',fontsize=10)
fig.savefig(root/'geometry.png',dpi=150);plt.close(fig)
print(json.dumps({k:[dict(goal=x['goal'],geodesic=x['geodesic'],path=x['path']) for x in result[k]] for k in ['v3','v4']},indent=2));print(result['v4_reflection_check'])
