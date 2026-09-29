import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent/'report'
records=json.loads((out/'wandb_final_summary.json').read_text())['records']
values=np.array([[r['success_rate'] for r in records if r['task']==t]
                 for t in ('v1','v2','v3','v4')])*100
fig,ax=plt.subplots(figsize=(7,5.8))
im=ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=100)
for i in range(4):
    for j in range(4):
        ax.text(j,i,f'{values[i,j]:.0f}%',ha='center',va='center',
                color='white' if values[i,j]>60 else '#172b3a',fontsize=16)
ax.set_xticks(range(4),['+0.1','+0.5','+0.7','+0.9'])
ax.set_yticks(range(4),['v1','v2','v3','v4'])
ax.set_xlabel('DACER entropy target per action dimension');ax.set_ylabel('Maze')
ax.set_title('Final direct-policy goal success | n=100/run\nT1, regulator interval500, seed0 | 500k post-warmup',fontsize=12)
fig.colorbar(im,ax=ax,label='Goal success (%)',shrink=.75)
fig.tight_layout(rect=(0,.08,1,1))
fig.text(.5,.035,'W&B final summaries, all16 finished.\nSuccess rate alone does not establish multiple routes.',ha='center',fontsize=9)
fig.savefig(out/'final_success_wandb.png',dpi=170);plt.close(fig)
