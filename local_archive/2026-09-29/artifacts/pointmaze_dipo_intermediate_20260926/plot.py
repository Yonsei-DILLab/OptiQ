"""Plot six preserved DIPO evaluation arrays without retraining."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from maze_benchmarks.visualize_pointmaze import plot_map, plot_rollouts

root=Path(__file__).resolve().parent
fig, axes=plt.subplots(3,2,figsize=(12,16),constrained_layout=True)
files={}
for row,maze in enumerate(('simple','medium','hard')):
    folder=root/maze
    record=json.loads((folder/'000600064_summary.json').read_text())
    if record['step']!=600064 or record['task']!=f'pm_{maze}' or record['method']!='dipo':
        raise ValueError('wrong evaluation')
    for col,(mode,obstacle) in enumerate((('policy',False),('obstacle_policy',True))):
        path=folder/f'000600064_{mode}.npz'
        summary=record[mode]
        with np.load(path) as raw:
            ids=raw['goal_ids']
            counts=np.bincount(ids[ids>=0],minlength=len(summary['goals'])).tolist()
            if (counts!=summary['goals'] or len(ids)!=summary['episodes'] or
                    int((ids<0).sum())!=summary['failure']):
                raise ValueError('raw/summary disagreement')
        ax=axes[row,col]
        plot_map(ax,maze,obstacle=obstacle)
        plot_rollouts(ax,path,max_trajectories=100)
        ax.set_title(f"{maze.title()} · {'obstacles' if obstacle else 'original map'}\n"
                     f"success {summary['success']:.0%} · goals {counts}",fontsize=11)
        files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    files[str((folder/'000600064_summary.json').relative_to(root))]=hashlib.sha256((folder/'000600064_summary.json').read_bytes()).hexdigest()
fig.suptitle('DIPO PointMaze · 600,064 transitions · seed 0 · native policy sampling\n'
             'First 100 trajectories shown in each panel; statistics use all 200 episodes',fontsize=15)
fig.savefig(root/'dipo_600k_trajectories.png',dpi=175,bbox_inches='tight')
plt.close(fig)
(root/'provenance.json').write_text(json.dumps(dict(training_source=record['source_commit'],
  reporting_source='fa0767e6933402506966269744b3834b2e0ee9e8',
  evaluation_mode='DIPO native policy',step=600064,episodes=200,
  files_sha256=files,training_modified=False),indent=2)+'\n')
print(root/'dipo_600k_trajectories.png')
