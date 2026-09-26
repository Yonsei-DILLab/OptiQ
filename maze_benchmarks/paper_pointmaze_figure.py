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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .visualize_pointmaze import get_map, plot_map, plot_rollouts
from .plot_style import (SUCCESS_TRAJECTORY_COLOR, FAILURE_TRAJECTORY_COLOR,
                         VISITED_GOAL_COLOR, UNVISITED_GOAL_COLOR, TRAJECTORY_PALETTES)


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
    cells=get_map(maze);height=len(cells);width=len(cells[0])
    positions=[[col+.5-width/2,height/2-row-.5]
               for row,values in enumerate(cells) for col,value in enumerate(values) if value=='g']
    if not np.array_equal(positions,config['geometry']['goal_positions']):
        raise ValueError('rendered map goal order differs from saved goal IDs')
    if np.any(ids < -1) or np.any(ids >= n):raise ValueError('invalid saved goal ID')
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


def build(root, temps, out, columns, dipo_root=None, alpha=.5,
          outcome_colors=False, formats=('pdf','png','svg'), trajectory_palette='mint-pink',
          match_goal_colors=False):
    success_color,failure_color=TRAJECTORY_PALETTES[trajectory_palette]
    reached_goal_color=success_color if match_goal_colors else VISITED_GOAL_COLOR
    unreached_goal_color=failure_color if match_goal_colors else UNVISITED_GOAL_COLOR
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
            entry=load(root,maze,method,temps,dipo_root)
            ax=fig.add_axes([left+col*(w+gap),bottoms[row],w,h])
            plot_map(ax,maze,goal_counts=None if entry is None else entry['counts'],
                     outcome_colors=outcome_colors,visited_goal_color=reached_goal_color,
                     unvisited_goal_color=unreached_goal_color)
            if entry is None:
                pending.append(dict(method=method,maze=maze))
                # No synthetic or earlier-budget trajectories in a final-result slot.
                ax.text(.5,.5,'1M result\npending',ha='center',va='center',transform=ax.transAxes,
                    fontsize=13,color='#505050',zorder=20,
                    bbox=dict(facecolor='white',alpha=.97,edgecolor='none',pad=8))
            else:
                before=len(ax.lines)
                plot_rollouts(ax,Path(entry['raw']),max_trajectories=500,alpha=alpha,
                              outcome_colors=outcome_colors,success_color=success_color,
                              failure_color=failure_color)
                trajectories=ax.lines[before:]
                if len(trajectories)!=500 or any(x.get_alpha()!=alpha for x in trajectories):
                    raise ValueError('rendered rollout count/alpha differs')
                if outcome_colors:
                    with np.load(entry['raw']) as raw:
                        expected_colors=[success_color if g>=0 else failure_color
                                         for g in raw['goal_ids']]
                    if [line.get_color() for line in trajectories]!=expected_colors:
                        raise ValueError('trajectory outcome colors do not match saved goal IDs')
                data.append(entry)
            ax.set(xlabel='',ylabel='',xticks=[],yticks=[])
            for spine in ax.spines.values():spine.set_visible(False)
    for row,maze in enumerate(('Medium','Hard')):
        fig.text(.014,bottoms[row]+h/2,maze,rotation=90,ha='center',va='center',fontsize=14,fontweight='normal')
    if outcome_colors:
        handles=[Line2D([0],[0],color=success_color,lw=2,label='Successful trajectory'),
                 Line2D([0],[0],color=failure_color,lw=2,label='Failed trajectory'),
                 Patch(facecolor=reached_goal_color,label='Goal reached (at least once)'),
                 Patch(facecolor=unreached_goal_color,label='Goal not reached')]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(.514,.025),ncol=4,
                   frameon=False,fontsize=10,handlelength=2,columnspacing=1.6)
    fig.text(.514,-.045 if outcome_colors else .05,'Figure 4: PointMaze.',
             fontproperties=caption_font,ha='center',va='center')
    suffix='' if alpha==.5 else '_alpha'+f'{alpha:g}'.replace('.','')
    name=f'figure4_pointmaze_{columns}methods'+suffix+('_outcomes' if outcome_colors else '')+('_draft' if pending else '')
    out.mkdir(parents=True,exist_ok=True)
    for extension in formats:
        fig.savefig(out/f'{name}.{extension}',dpi=350,facecolor='white',bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    manifest=dict(figure=name,methods=methods,rows=['medium','hard'],pending=pending,
        figure_is_complete=not pending,inputs=data,rollouts_per_panel=500,alpha=alpha,linewidth=1.8,
        algorithm_font='Arial sans-serif; only iBOLT uses Arial Bold',
        title='Figure4: PointMaze',training_seed=0,ibolt_temperature=5.,
        ibolt_sampling='fresh normal z each action; mu-only; conditional sigma off',
        baseline_sampling='native policy sampling; original entropy settings, not the new sensitivity grid',
        outcome_colors=outcome_colors,
        trajectory_palette=trajectory_palette if outcome_colors else None,
        goal_colors_match_trajectories=match_goal_colors if outcome_colors else None,
        color_semantics=dict(success=success_color,failure=failure_color,
                             reached_goal=reached_goal_color,unreached_goal=unreached_goal_color,
                             goal_rule='at least one terminal success in the 500 displayed episodes',
                             pending_goals='unknown, neutral gray',draw_order='original episode order')
                        if outcome_colors else None,
        reporting_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        caveat='A goal observed once counts as visited; this is not balanced multimodal coverage.',
        outputs={ext:sha(out/f'{name}.{ext}') for ext in formats})
    (out/f'{name}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--temperatures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dipo-root',type=Path)
    p.add_argument('--alphas',type=float,nargs='+',default=[.5])
    p.add_argument('--outcome-colors',action='store_true')
    p.add_argument('--trajectory-palette',choices=TRAJECTORY_PALETTES,default='mint-pink')
    p.add_argument('--match-goal-colors',action='store_true')
    p.add_argument('--formats',choices=('png','pdf','svg'),nargs='+',default=['pdf','png','svg'])
    a=p.parse_args()
    if any(not np.isfinite(v) or not 0<v<=1 for v in a.alphas):raise ValueError('invalid alpha')
    for alpha in a.alphas:
        for count in (6,7):
            m=build(a.root,a.temperatures,a.output,count,a.dipo_root,alpha=alpha,
                    outcome_colors=a.outcome_colors,formats=a.formats,
                    trajectory_palette=a.trajectory_palette,match_goal_colors=a.match_goal_colors)
            print(json.dumps(dict(figure=m['figure'],complete=m['figure_is_complete'],pending=m['pending'])))
    (a.output/'CAPTION.txt').write_text('Figure4: PointMaze. Top: Medium; bottom: Hard. Each completed panel shows all500 evaluation trajectories from one seed0 policy after1M environment interactions. iBOLT uses T=5 and fresh Gaussian latent z at each action with mu-only output (no conditional sigma). Baselines use their native sampling rules and original entropy settings. The seven-column draft reserves DIPO until the official implementation completes1M; no older variant is substituted.\n')
    if a.outcome_colors:
        success_name,failure_name=a.trajectory_palette.split('-')
        reached_name=success_name if a.match_goal_colors else 'dark blue'
        unreached_name=failure_name if a.match_goal_colors else 'dark red'
        with (a.output/'CAPTION.txt').open('a') as caption:
            caption.write(f'{success_name.capitalize()} trajectories end in a recorded terminal goal success; {failure_name} trajectories do not. {reached_name.capitalize()} goals were reached at least once in the same500 episodes; {unreached_name} goals were never reached. Pending panels use neutral gray goals. The black dot marks the start. Episode draw order is unchanged.\n')


if __name__=='__main__':main()
