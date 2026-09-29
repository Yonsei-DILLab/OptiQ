import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('verified_v1_report',root/'report_results.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
data,_=report.collect()
names=('step2m75-r2','final-3m-seed20260925','final-independent')
assert all((name,mode) in data for name in names for mode in ('policy','native'))
walls,goals,bounds=report.maze_geometry('v1')
fig,axes=plt.subplots(2,3,figsize=(12,9),layout='constrained')
for col,name in enumerate(names):
    for row,mode in enumerate(('policy','native')):
        ax=axes[row,col];e=data[name,mode];r=e['row']
        for x0,y0,x1,y1 in walls:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#e2e6e9',edgecolor='#bdc4cb',lw=.4))
        for i in np.argsort(e['goals']>0):
            p=e['points'][i];success=e['goals'][i]>0
            ax.plot(p[:,0],p[:,1],color=report.COLORS[e['labels'][i]],alpha=.4 if success else .13,lw=.9 if success else .6)
        title=report.NAMES[name]+'\n'+('Direct policy: random z + sigma' if mode=='policy' else 'Random-z mu-only')
        title+=f"\nSuccess U:{r['successful_routes'].get('upper',0)} D:{r['successful_routes'].get('lower',0)} | failed:{r['failures']}"
        ax.set_title(title,fontsize=9)
        ax.scatter(*goals.T,marker='*',s=95,c='#38a35f',edgecolor='white',lw=.6)
        ax.scatter(0,0,s=20,c='black',zorder=7)
        ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
fig.suptitle('Official AntMaze v1: two successful routes from identical full initial state\nT3 | dense = -nearest-goal distance | DACER OFF | one training seed | n=100 per panel\nSupplementary central-start evaluation. Earlier2M/2.5M failures remain in the full report.',fontsize=12)
fig.savefig(root/'report/late_same_state_routes.png',dpi=165);plt.close(fig)
