"""Two-row paper figures with verified500-rollout inputs and bold iBOLT(T5)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
import numpy as np

from .visualize_pointmaze import plot_map, plot_rollouts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, maze, method, temperature_root, dipo_root=None):
    if method == 'optiq':
        base=temperature_root;name=f'pm_{maze}-optiq-t5-s0'
    elif method == 'dipo':
        if dipo_root is None:return None
        base=dipo_root;name=f'pm_{maze}-dipo-s0'
    else:
        base=root;name=f'pm_{maze}-{method}-s0'
    folder=base/'runs'/name
    if method=='dipo' and not (folder/'progress.json').exists():return None
    config=json.loads((folder/'config.json').read_text())
    progress=json.loads((folder/'progress.json').read_text())
    if method=='dipo':
        if config['agent'].get('upstream')!='BellmanTimeHut/DIPO':
            raise ValueError('Do not relabel the historical DDiffPG DIPO variant as official DIPO')
        if progress['status']!='complete':return None
    if (progress['status'],progress['steps'],progress['updates'])!=('complete',1000192,62000):
        raise ValueError('paper plate requires completed matched1M runs')
    mode='mu_only' if method=='optiq' else 'policy'
    if config['task']!=f'pm_{maze}' or config['method']!=method or config['seed']!=0:
        raise ValueError('wrong method/maze/seed')
    if method=='optiq' and config['temperature']!=5.:raise ValueError('iBOLT must useT5')
    raw=folder/'evaluations'/f'001000192_{mode}.npz'
    with np.load(raw) as z:
        ids=z['goal_ids'];tracks=z['xy'];returns=z['returns']
    if len(ids)!=500 or tracks.shape[0]!=500 or not np.isfinite(returns).all():
        raise ValueError('not500 valid evaluation episodes')
    n=8 if maze=='hard' else 4
    counts=np.bincount(ids[ids>=0],minlength=n).tolist()
    expected=progress['latest_evaluation'][mode]
    if counts!=expected['goals'] or int(np.sum(ids<0))!=expected['failure']:
        raise ValueError('raw and summary mismatch')
    archive=json.loads((base/'archive-manifest.json').read_text())['runs'][name]
    relative=str(raw.relative_to(folder));actual_sha=sha(raw)
    if archive['sha256'][relative]!=actual_sha:raise ValueError('unverified raw evaluation')
    return dict(raw=str(raw),sha256=actual_sha,source=config['source_commit'],run=name,
                task=config['task'],method=method,mode=mode,temperature=config.get('temperature'),
                episodes=500,counts=counts,failure=int(np.sum(ids<0)),
                success=float(np.mean(ids>=0)),steps=progress['steps'],updates=progress['updates'])


def build(root, temps, out, columns, dipo_root=None, alpha=.5):
    methods=['optiq','sac','sql','meow','mfpo']+(['dipo'] if columns==7 else [])+['td3']
    regular=FontProperties(family='Arial',size=15)
    bold=FontProperties(family='Arial',weight='bold',size=15)
    caption_font=FontProperties(family='Times New Roman',size=19)
    plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path'})
    width=columns*2.18+.46;height=5.25
    fig=plt.figure(figsize=(width,height),facecolor='white')
    left=.035;right=.993;gap=.012
    w=(right-left-gap*(columns-1))/columns
    h=w*width/height
    bottoms=[.895-h,.865-2*h]
    data=[];pending=[]
    for col,method in enumerate(methods):
        center=left+col*(w+gap)+w/2
        label='iBOLT' if method=='optiq' else method.upper()
        parts=[TextArea(f'({chr(97+col)}) ',textprops={'fontproperties':regular}),
               TextArea(label,textprops={'fontproperties':bold if method=='optiq' else regular})]
        title=AnchoredOffsetbox(loc='lower center',child=HPacker(children=parts,align='baseline',pad=0,sep=0),
            frameon=False,pad=0,bbox_to_anchor=(center,.925),bbox_transform=fig.transFigure,borderpad=0)
        fig.add_artist(title)
        for row,maze in enumerate(('medium','hard')):
            ax=fig.add_axes([left+col*(w+gap),bottoms[row],w,h]);plot_map(ax,maze)
            entry=load(root,maze,method,temps,dipo_root)
            if entry is None:
                pending.append(dict(method=method,maze=maze))
                # No synthetic or earlier-budget trajectories in a final-result slot.
                ax.text(.5,.5,'1M result\npending',ha='center',va='center',transform=ax.transAxes,
                    fontsize=13,color='#505050',zorder=20,
                    bbox=dict(facecolor='white',alpha=.97,edgecolor='none',pad=8))
            else:
                before=len(ax.lines);plot_rollouts(ax,Path(entry['raw']),max_trajectories=500,alpha=alpha)
                trajectories=ax.lines[before:]
                if len(trajectories)!=500 or any(x.get_alpha()!=alpha for x in trajectories):
                    raise ValueError('rendered rollout count/alpha differs')
                data.append(entry)
            ax.set(xlabel='',ylabel='',xticks=[],yticks=[])
            for spine in ax.spines.values():spine.set_visible(False)
    for row,maze in enumerate(('Medium','Hard')):
        fig.text(.014,bottoms[row]+h/2,maze,rotation=90,ha='center',va='center',fontsize=14,fontweight='normal')
    fig.text(.514,.05,'Figure 4: PointMaze.',fontproperties=caption_font,ha='center',va='center')
    suffix='' if alpha==.5 else '_alpha'+f'{alpha:g}'.replace('.','')
    name=f'figure4_pointmaze_{columns}methods'+suffix+('_draft' if pending else '')
    out.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','png','svg'):
        fig.savefig(out/f'{name}.{extension}',dpi=350,facecolor='white',bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    manifest=dict(figure=name,methods=methods,rows=['medium','hard'],pending=pending,
        figure_is_complete=not pending,inputs=data,rollouts_per_panel=500,alpha=alpha,linewidth=1.8,
        algorithm_font='Arial sans-serif; only iBOLT uses Arial Bold',
        title='Figure4: PointMaze',training_seed=0,ibolt_temperature=5.,
        ibolt_sampling='fresh normal z each action; mu-only; conditional sigma off',
        baseline_sampling='native policy sampling; original entropy settings, not the new sensitivity grid',
        reporting_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        caveat='A goal observed once counts as visited; this is not balanced multimodal coverage.',
        outputs={ext:sha(out/f'{name}.{ext}') for ext in ('pdf','png','svg')})
    (out/f'{name}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--temperatures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dipo-root',type=Path)
    p.add_argument('--alphas',type=float,nargs='+',default=[.5])
    a=p.parse_args()
    if any(not np.isfinite(v) or not 0<v<=1 for v in a.alphas):raise ValueError('invalid alpha')
    for alpha in a.alphas:
        for count in (6,7):
            m=build(a.root,a.temperatures,a.output,count,a.dipo_root,alpha=alpha)
            print(json.dumps(dict(figure=m['figure'],complete=m['figure_is_complete'],pending=m['pending'])))
    (a.output/'CAPTION.txt').write_text('Figure4: PointMaze. Top: Medium; bottom: Hard. Each completed panel shows all500 evaluation trajectories from one seed0 policy after1M environment interactions. iBOLT uses T=5 and fresh Gaussian latent z at each action with mu-only output (no conditional sigma). Baselines use their native sampling rules and original entropy settings. The seven-column draft reserves DIPO until the official implementation completes1M; no older variant is substituted.\n')


if __name__=='__main__':main()
