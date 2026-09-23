"""Inspect saved policies separately from behavior exploration, including failures."""
import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle,Circle
import numpy as np

METHODS=['optiq','sac','dipo','mfpo']
LABELS=dict(optiq='OptiQ',sac='SAC',dipo='DIPO',mfpo='MFPO')
COLORS=dict(optiq='#197ab3',sac='#b77924',dipo='#178970',mfpo='#9654aa')
GOALS=dict(v1=[[-8,0]],v2=[[-8,8],[8,0]],v3=[[-12,12],[12,-12]],v4=[[-16,4],[-16,-4]])
REPO=Path(__file__).resolve().parents[1]


def load(path):return json.loads(path.read_text())


class Markers(ast.NodeTransformer):
    def visit_Name(self,node):return ast.Constant({'R':'r','G':'g'}[node.id])


tree=ast.parse((REPO/'antmaze/ddiffpg/env/d4rl/locomotion/maze_env.py').read_text())
MAPS={}
for node in tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name in ['MAZE_'+t for t in GOALS]:MAPS[name[5:]]=ast.literal_eval(Markers().visit(node.value))


def geometry(task):
    maze=MAPS[task]
    rr,cc=next((i,j) for i,row in enumerate(maze) for j,v in enumerate(row) if v=='r')
    return maze,rr,cc


def decorate(ax,task,start):
    maze,rr,cc=geometry(task)
    for i,row in enumerate(maze):
        for j,cell in enumerate(row):
            if cell==1:ax.add_patch(Rectangle(((j-cc)*4-2,(i-rr)*4-2),4,4,
                        facecolor='#e1e6ec',edgecolor='#c5ccd3',lw=.6,zorder=0))
    for g in GOALS[task]:
        ax.add_patch(Circle(g,.5,color='#d9970c',alpha=.3,zorder=5))
        ax.scatter(*g,marker='*',s=100,c='#e7a924',edgecolor='#825c16',lw=.4,zorder=6)
    ax.scatter(*start,marker='^',s=30,c='#172936',zorder=7)
    ax.set_xlim(-cc*4-2,(len(maze[0])-1-cc)*4+2)
    ax.set_ylim((len(maze)-1-rr)*4+2,-rr*4-2)
    ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')


def rollout(path,task):
    with np.load(path,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
    xy,n,goals=d['xy'],d['lengths'],d['goals']
    assert len(xy)==len(n)==len(goals)==100
    for line,length in zip(xy,n):
        assert np.isfinite(line[:length+1]).all() and np.isnan(line[length+1:]).all()
    distance=np.linalg.norm(xy[:,:,None,:]-np.array(GOALS[task])[None,None,:,:],axis=-1)
    reached=np.nanmin(distance,axis=(1,2))<=.50002
    assert np.array_equal(reached,goals>0)
    expected=-np.nansum(distance[:,1:,:].min(axis=-1),axis=1)
    assert np.allclose(expected,d['returns'],rtol=2e-6,atol=.003),path
    if bool(d['fixed']):assert np.all(d['initial_full_state']==d['initial_full_state'][0])
    points=xy[np.isfinite(xy).all(-1)]
    displacement=np.linalg.norm(xy-xy[:,:1],axis=-1)
    metrics=dict(successes=int(reached.sum()),episodes=len(goals),mean_return=float(d['returns'].mean()),
        goal_counts={str(g):int((goals==g).sum()) for g in np.unique(goals)},
        closest_goal_m=float(np.nanmin(distance)),
        median_max_displacement_m=float(np.median(np.nanmax(displacement,axis=1))),
        bins_05m=len(np.unique(np.floor(points/.5).astype(np.int32),axis=0)))
    if task=='v1':metrics.update(
        upper_entries=int(np.any((xy[:,:,0]<-2)&(xy[:,:,1]<-2),axis=1).sum()),
        lower_entries=int(np.any((xy[:,:,0]<-2)&(xy[:,:,1]>2),axis=1).sum()))
    else:metrics.update(left_entries=int(np.any(xy[:,:,0]<-4,axis=1).sum()),
                        right_entries=int(np.any(xy[:,:,0]>4,axis=1).sum()))
    return d,metrics


def report(root):
    out=root/'report';out.mkdir(exist_ok=True)
    rows={};data={};sources=set();tasks=set();steps=set();audits={}
    for host in ('vast-heechan-180','vast-heechan-199'):
        assert load(root/host/'archive-verification.json')['verified']
        for run in sorted((root/host/'runs').iterdir()):
            cfg=load(run/'config.json');result=load(run/'result.json')
            task,method=cfg['task'],cfg['method'];key=run.name
            sources.add(cfg['source_commit']);tasks.add(task);steps.add(cfg['steps'])
            assert cfg['reward_profile']=='dense' and cfg['noveld_enabled'] is False
            assert cfg['num_envs']==256 and cfg['batch_size']==4096
            assert result['completed'] and result['rnd_updates']==0
            assert result['updates']==cfg['expected_updates']
            audit=load(run/'parameter-audit.json');assert audit['passed']
            assert 'rnd_predictor' not in audit['final'] and 'rnd_target' not in audit['final']
            audits[(task,method)]=audit['initial']
            xy=np.load(run/'training-xy.npy',mmap_mode='r')
            assert xy.shape==(cfg['steps'],2) and np.isfinite(xy).all()
            cells,first,counts=np.unique(np.floor(xy/.5).astype(np.int32),axis=0,
                                         return_index=True,return_counts=True)
            successes=load(run/'training-successes.json')
            assert len(successes)==result['training_successes']
            training=dict(bins_05m=len(cells),successes=len(successes),
                first_success_step=min((r['step'] for r in successes),default=None),
                closest_goal_m=float(np.linalg.norm(xy[:,None,:]-np.array(GOALS[task])[None,:,:],axis=-1).min()),
                top10_bin_fraction=float(np.sort(counts)[-10:].sum()/len(xy)))
            modes={}
            for label in ('policy-fixed','native-fixed','policy-natural','native-natural'):
                modes[label]=rollout(run/'evaluations'/f'{cfg["steps"]:010d}'/label/'rollouts.npz',task)
            rows[key]=dict(task=task,method=method,host=host,steps=cfg['steps'],updates=result['updates'],
                training=training,modes={label:m for label,(_,m) in modes.items()})
            data[key]=dict(xy=xy,cells=cells,first=first,counts=counts,modes=modes)
    assert len(sources)==1
    assert steps=={328192} and len(rows)==8, 'This report is the eight-policy 10k-update probe'
    tasks=sorted(tasks)
    for task in tasks:
        starts=[data[f'{task}-{m}-s0']['modes']['policy-fixed'][0]['initial_full_state'][0] for m in METHODS]
        assert all(np.array_equal(starts[0],s) for s in starts)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.facecolor':'white',
                         'axes.spines.top':False,'axes.spines.right':False})
    for label,filename,title in [('policy-fixed','policy_fixed_trajectories.png','Direct policy sampling'),
                                ('policy-natural','policy_natural_trajectories.png','Direct policy / original reset distribution'),
                                ('native-fixed','native_fixed_trajectories.png','Native evaluation controls')]:
        fig,axes=plt.subplots(len(tasks),4,figsize=(15,4.8*len(tasks)),squeeze=False,constrained_layout=True)
        for row,task in enumerate(tasks):
            for col,method in enumerate(METHODS):
                ax=axes[row,col];d,m=data[f'{task}-{method}-s0']['modes'][label]
                for line,length,goal in zip(d['xy'],d['lengths'],d['goals']):
                    line=line[:length+1]
                    ax.plot(line[:,0],line[:,1],color=COLORS[method],alpha=.24,lw=.55,zorder=2)
                    ax.scatter(*line[-1],s=5,c='#289350' if goal else '#d85d4f',alpha=.4,zorder=3)
                decorate(ax,task,d['xy'][0,0])
                if label=='policy-natural':
                    ax.scatter(d['xy'][:,0,0],d['xy'][:,0,1],marker='^',s=9,
                               c='#172936',alpha=.3,zorder=7)
                ax.set_title(f'{task.upper()} | {LABELS[method]} | success {m["successes"]}/100\n'
                             f'closest goal {m["closest_goal_m"]:.2f} m; displacement {m["median_max_displacement_m"]:.1f} m')
        reset_title=('100 rollouts; v1 random starts, v3 original fixed start' if label=='policy-natural' else
                     '100 identical-state rollouts; v1 start (1.99, -1.45), v3 start (0, 0)')
        fig.suptitle(f'Dense reward + NovelD OFF | {title} | seed 0\n'
                     f'10k learner updates / 328,192 transitions; {reset_title}',fontsize=13)
        extra=('OptiQ includes conditional sigma; no external exploration noise. '
               if label.startswith('policy-') else 'OptiQ random-z mu-only; SAC mean; MFPO Q-best-of10; DIPO native diffusion. ')
        fig.supxlabel(extra+'Stars: goals; red dots: failed endpoints. All failures shown.',fontsize=9)
        fig.savefig(out/filename,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(len(tasks),4,figsize=(15,4.8*len(tasks)),squeeze=False,constrained_layout=True)
    for row,task in enumerate(tasks):
        maze,rr,cc=geometry(task)
        xe=np.arange(-cc*4-2,(len(maze[0])-1-cc)*4+2+.25,.5)
        ye=np.arange(-rr*4-2,(len(maze)-1-rr)*4+2+.25,.5)
        for col,method in enumerate(METHODS):
            key=f'{task}-{method}-s0';ax=axes[row,col];xy=data[key]['xy'];tr=rows[key]['training']
            grid,*_=np.histogram2d(xy[:,0],xy[:,1],bins=(xe,ye))
            mesh=ax.pcolormesh(xe,ye,np.ma.masked_equal((grid/len(xy)).T,0),cmap='magma',
                             norm=LogNorm(vmin=1/328192,vmax=.1),zorder=2)
            decorate(ax,task,(0,0))
            ax.set_title(f'{task.upper()} | {LABELS[method]} | {tr["bins_05m"]} bins\n'
                         f'train successes {tr["successes"]}; nearest goal {tr["closest_goal_m"]:.2f} m')
    fig.colorbar(mesh,ax=axes,label='Fraction of training visits per 0.5 m bin',shrink=.8)
    fig.suptitle('Behavior exploration during training | dense reward + NovelD OFF\n'
                 '328,192 transitions; exploration noise included; separate from final policy rollouts',fontsize=14)
    fig.savefig(out/'training_occupancy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,len(tasks),figsize=(6*len(tasks),4),squeeze=False,constrained_layout=True)
    for ax,task in zip(axes[0],tasks):
        for method in METHODS:
            first=np.sort(data[f'{task}-{method}-s0']['first'])
            ax.step(np.r_[0,first+1,328192]/1000,np.r_[0,np.arange(1,len(first)+1),len(first)],
                    where='post',label=LABELS[method],color=COLORS[method])
        ax.set_title(task.upper());ax.set_xlabel('Environment transitions (k)');ax.set_ylabel('Visited 0.5 m bins')
        ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'coverage_curves.png',dpi=180);plt.close(fig)
    summary=dict(source_commit=next(iter(sources)),single_seed=0,runs=rows,
                 primary='direct-policy, identical-full-state, 100 episodes; no exploration noise',
                 natural_fixed_not_pooled=True,technical_validation_passed=True)
    (out/'results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    lines=['# Dense + NovelD OFF: 10k learner update 검증','',
           '각 방법·미로 단일 seed0. 256env, batch4096, 총328,192 transitions. 학습 중 탐색과 최종 정책을 구분한다.',
           '주 그림은 같은 simulator full state에서 직접 정책 샘플링한100회. 실패 궤적도 모두 포함했다.',
           '', '|미로|방법|학습 방문격자(0.5m)|학습 도달|고정시작 정책 성공/100|정책 최근접 목표(m)|',
           '|---|---|---:|---:|---:|---:|']
    for task in tasks:
        for method in METHODS:
            r=rows[f'{task}-{method}-s0'];tr=r['training'];m=r['modes']['policy-fixed']
            lines.append(f'|{task}|{LABELS[method]}|{tr["bins_05m"]}|{tr["successes"]}|{m["successes"]}|{m["closest_goal_m"]:.2f}|')
    lines+=['','DIPO: float64 C51 projection·확률 경계·gradient norm1·finite guard 적용. Dense 호환 support[-6000,5],51atoms.',
            '모든 평가 return을 저장된 매 step 위치의 실제 dense 거리합과 대조했다. NovelD/RND update는 모두0이다.',
            '체크포인트는 원격 보존하고 SHA256를 다시 확인했다. 원시 궤적·학습xy·config·검증 메타데이터는 로컬 보관.',
            '성공이나 다양한 성공 경로를 얻었다는 결론은 실제 도달 수와 경로 확인으로만 판단한다. 짧은 학습·단일seed의 한계가 있다.',
            f'학습 source: `{next(iter(sources))}`.']
    lines+=['','## v1 시작 위치 영향',
            '원본 v1은 매 reset에서 x,y를 각각[-2,2]에서 뽑는다. 고정 시작 그림은 그중(1.9873,-1.4523)을100회 반복한 조건부 평가다.',
            '이 시작점은 위쪽에 치우쳐 있으므로 이 그림 하나로 전체 정책의 경로 편중을 단정하지 않는다.',
            'policy_natural_trajectories.png는 원래 랜덤 시작 분포를 사용한다. 여러 시작에서 두 통로를 가는 결과도 같은 상태의 다봉 정책을 증명하지는 않는다.',
            '', '|방법|고정 시작 위/아래 통로 진입|랜덤 시작 위/아래 통로 진입|랜덤 시작 성공/100|',
            '|---|---:|---:|---:|']
    for method in METHODS:
        a=rows[f'v1-{method}-s0']['modes']['policy-fixed'];b=rows[f'v1-{method}-s0']['modes']['policy-natural']
        lines.append(f'|{LABELS[method]}|{a["upper_entries"]}/{a["lower_entries"]}|{b["upper_entries"]}/{b["lower_entries"]}|{b["successes"]}|')
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(training=v['training'],policy=v['modes']['policy-fixed']) for k,v in rows.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)
