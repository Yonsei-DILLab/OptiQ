"""Collect and audit four 100k runs; plot successful and failed rollouts together."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from antmaze.multimodal.dense_noveld_report import read,write,verify_run,maze,COLORS
NAME='antmaze-v3-optiq-noveld-strength-100k-s0-20260922'
PROFILE='dense-noveld-strength-100k'
COEFFICIENTS=(.1,1.,5.,10.)


def job_id(coefficient):return 'v3-optiq-c'+format(coefficient,'g').replace('.','p')+'-s0'

TRAINING_SHA='262a10d6280eb6e79eec24e6170541a7b0f58e72'
DEFAULT=Path('artifacts/antmaze_noveld_strength_100k')


def collect(root):
    root.mkdir(parents=True,exist_ok=True)
    remote=f'vast-heechan-180:/home/heechan/optiq-experiments/{NAME}/'
    subprocess.run(['rsync','-az','--exclude=preflight/','--exclude=wandb/','--exclude=controller.lock',remote,str(root)+'/'],check=True)
    status=read(root/'status.json')
    assert status['source_commit']==TRAINING_SHA
    write(root/'collection.json',dict(host='vast-heechan-180',remote=remote,training_source=TRAINING_SHA,
        reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for j in status['jobs']:
        folder=root/'runs'/j['id'];p=folder/'progress.json'
        print(j['id'],j['status'],read(p).get('env_steps') if p.exists() else None)
    return status


def logged_metrics(root,job):
    rows=[]
    for line in (root/'logs'/f'training-{job}.log').read_text().splitlines():
        if not line.startswith('{'):continue
        try:d=json.loads(line)
        except json.JSONDecodeError:continue
        if 'intrinsic_mean' in d:rows.append(d)
    return rows


def audit(root):
    manifest=read(root/'manifest.json');assert manifest['source_commit']==TRAINING_SHA
    assert read(root/'result.json')['completed']
    assert read(root/'training-matched-initialization.json')['passed']
    configs=[];rows=[]
    for coefficient in COEFFICIENTS:
        job=job_id(coefficient);folder=root/'runs'/job
        proof=verify_run(folder,expected_source=TRAINING_SHA,expected_steps=100000,
            expected_profile=PROFILE,expected_coefficient=coefficient)
        write(folder/'archive-verification.json',proof)
        c=read(folder/'config.json');r=read(folder/'result.json');configs.append(c)
        with np.load(folder/'resume'/'step_0000100000'/'replay.npz') as z:
            xy=z['next_observations'][:,:2];done=z['dones'].astype(bool)
        dist=np.linalg.norm(xy[:,None,:]-np.asarray(c['environment']['goals'])[None,:,:],axis=-1)
        reached=dist.argmin(axis=1)+1
        first={};counts={}
        for goal in (1,2):
            steps=np.flatnonzero(done&(reached==goal))+1
            first[str(goal)]=int(steps[0]) if len(steps) else None;counts[str(goal)]=int(len(steps))
        coverage=np.load(folder/'training_coverage.npz')['counts']
        logs=logged_metrics(root,job);last=[d for d in logs if d['env_steps']>75000]
        row=dict(coefficient=coefficient,job=job,training_seed=0,steps=100000,updates=r['updates'],
            train_goal_counts=counts,train_first_goal_steps=first,train_min_goal_distances=dist.min(axis=0).tolist(),
            visited_bins=int(np.count_nonzero(coverage)),mean_logged_bonus_last25k=float(np.mean([v['intrinsic_mean'] for v in last])),
            mean_logged_env_reward_last25k=float(np.mean([v['env_reward_mean'] for v in last])),summaries=r['summaries'])
        rows.append(row)
    for label in ('policy-fixed','native-fixed'):
        starts=[]
        for coefficient in COEFFICIENTS:
            with np.load(root/'runs'/job_id(coefficient)/'rollouts'/f'100000-{label}.npz') as z:starts.append(z['initial_simulator_state'])
        for state in starts[1:]:np.testing.assert_array_equal(starts[0],state)
    write(root/'report'/'results.json',dict(training_source=TRAINING_SHA,seed_count=1,training_seed=0,
        rollout_episodes_per_panel=100,steps=100000,shared_fixed_start_verified=True,results=rows))
    return configs,rows


def render(root,configs,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm
    out=root/'report';out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    for reset in ('fixed','natural'):
        fig,axes=plt.subplots(2,4,figsize=(14,8.3))
        for ri,mode in enumerate(('native','policy')):
            for ci,(c,row) in enumerate(zip(configs,rows)):
                ax=axes[ri,ci];maze(ax,c);s=row['summaries'][mode+'-'+reset]
                z=np.load(root/'runs'/row['job']/'rollouts'/f'100000-{mode}-{reset}.npz')
                for xy,n,g in zip(z['xy'],z['lengths'],z['goal_ids']):
                    xy=xy[:int(n)+1];ax.plot(xy[:,0],xy[:,1],color=COLORS[int(g)],alpha=.25,lw=.7,zorder=3)
                g1=round(s['goal_fractions']['1']*100);g2=round(s['goal_fractions']['2']*100)
                ax.set_title(f"NovelD {row['coefficient']:g} | success {g1+g2}/100\nG1 {g1} | G2 {g2} | fail {100-g1-g2}",fontsize=10)
                if ci==0:ax.set_ylabel('Random-z mean (mu-only)' if mode=='native' else 'Random-z + conditional sigma',fontsize=10)
        fig.suptitle(f'AntMaze v3 · OptiQ · 100k interactions · training seed 0\n100 rollouts per panel · {reset} full initial state',fontsize=13)
        fig.legend(handles=[Line2D([0],[0],color=COLORS[g],label=l) for g,l in ((1,'Goal 1'),(2,'Goal 2'),(0,'Failed'))],
            loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.016))
        fig.text(.5,.003,'No external DACER noise or NovelD reward in evaluation. Each column is one independently trained policy.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.065,1,.925));save(fig,'trajectories-'+reset)
    fig,axes=plt.subplots(1,4,figsize=(14,4.4))
    maximum=max(np.load(root/'runs'/r['job']/'training_coverage.npz')['counts'].max() for r in rows)
    for ax,c,row in zip(axes,configs,rows):
        maze(ax,c);z=np.load(root/'runs'/row['job']/'training_coverage.npz');counts=z['counts'].astype(float);counts[counts==0]=np.nan
        im=ax.imshow(counts.T,origin='lower',extent=[z['low'][0],z['high'][0],z['low'][1],z['high'][1]],
            cmap='YlOrRd',norm=LogNorm(1,maximum),interpolation='nearest',zorder=1)
        ax.set_title(f"NovelD {row['coefficient']:g}\n{row['visited_bins']} visited bins",fontsize=10)
    fig.suptitle('Training coverage · same 100k budget · shared logarithmic color scale')
    fig.subplots_adjust(left=.02,right=.92,top=.84,bottom=.06,wspace=.12)
    fig.colorbar(im,cax=fig.add_axes([.94,.2,.012,.53]),label='Training visits');save(fig,'training-coverage')
    fig,axes=plt.subplots(2,3,figsize=(13,7.6))
    for ri,mode in enumerate(('native','policy')):
        for row in rows:
            h=read(root/'runs'/row['job']/f'history-{mode}-natural.json');x=[v['step']/1000 for v in h]
            for ci,key in enumerate(('success_rate','mean_min_distance','mean_return')):
                ax=axes[ri,ci];ax.plot(x,[v[key] for v in h],marker='o',label=f"coef {row['coefficient']:g}")
                ax.set(title=mode+' · '+key,xlabel='Environment interactions (k)');ax.grid(alpha=.2)
                if ci==0:ax.set_ylim(-.02,1.02)
        axes[ri,0].legend(fontsize=8)
    fig.suptitle('10 evaluation episodes per point · one training seed · no across-seed error band')
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'learning-curves')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for row in rows:
        logs=logged_metrics(root,row['job']);x=np.asarray([v['env_steps'] for v in logs])/1000
        for ci,key in enumerate(('intrinsic_mean','env_reward_mean','training_min_distance')):
            axes[ci].plot(x,[v[key] for v in logs],label=f"coef {row['coefficient']:g}")
            axes[ci].set(title=key,xlabel='Environment interactions (k)');axes[ci].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Actual training logs · bonus/env reward are minibatch snapshots every 1k steps')
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'reward-scale')
    lines=['# AntMaze v3 OptiQ NovelD coefficient: 100k',
        '',f'학습 소스: `{TRAINING_SHA}`. 각 계수당 학습 seed 0 하나. 100k 환경 step, 95k learner/RND update.',
        '기본값은 0.01을 유지하고 이번 네 실행만 계수 0.1/1/5/10을 적용했다. 나머지 설정 및 초기 모델/RND는 동일하다.',
        '', '고정된 동일 전체 초기 상태에서 각 정책 100회 평가. native는 random-z μ-only, policy는 random-z + conditional sigma다. 평가에 외부 DACER 잡음/NovelD 보상을 넣지 않는다.',
        'natural/fixed는 이 환경에서 같은 시작 분포이므로 합쳐 200개의 독립 초기 상태처럼 해석하지 않는다.',
        '', '| 계수 | 학습 G1/G2 도달 | 최초 G1/G2 step | 학습 최소 G1/G2 거리(m) | μ-only 성공 | full-policy 성공 | full-policy G1/G2 | 방문 bin |',
        '|---:|---:|---|---|---:|---:|---|---:|']
    for row in rows:
        p=row['summaries']['policy-fixed'];n=row['summaries']['native-fixed']
        counts=row['train_goal_counts'];first=row['train_first_goal_steps'];d=row['train_min_goal_distances']
        lines.append(f"| {row['coefficient']:g} | {counts['1']}/{counts['2']} | {first['1']}/{first['2']} | {d[0]:.2f}/{d[1]:.2f} | {n['success_rate']:.0%} | {p['success_rate']:.0%} | {p['goal_fractions']['1']:.0%}/{p['goal_fractions']['2']:.0%} | {row['visited_bins']} |")
    lines.extend(['','실패 궤적도 회색으로 전부 표시했다. 성공이 없는 경우 경로 다양성을 확보했다거나 mode collapse로 단정하지 않는다.',
        '100k는 초기 탐색 비교다. 단일 seed이며 이 예산에서 실패했다고 이후 학습 불가능을 의미하지 않는다.',
        '각 실행의 전체 100k replay 환경보상, checkpoint SHA256/full-state digest, 최종 600개 원시 rollout, 업데이트 수를 검증했다.',
        '', '## 실제 intrinsic 보너스 (마지막 25k, 매 1k 기록된 minibatch 평균들의 평균)',
        '', '| 계수 | intrinsic | environment |', '|---:|---:|---:|'])
    for row in rows:lines.append(f"| {row['coefficient']:g} | {row['mean_logged_bonus_last25k']:.4f} | {row['mean_logged_env_reward_last25k']:.4f} |")
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--local-only',action='store_true');a=p.parse_args()
    status=read(a.root/'status.json') if a.local_only else collect(a.root)
    if status['phase']=='completed':render(a.root,*audit(a.root));print('REPORT_READY',a.root/'report'/'REPORT_KO.md')
    elif (a.root/'failure.json').exists():raise RuntimeError(read(a.root/'failure.json'))


if __name__=='__main__':main()
