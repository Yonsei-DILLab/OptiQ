"""Read-only, equal-budget exploration comparison from verified saved replays."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from antmaze.multimodal.dense_noveld_report import maze, state_digest

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCES={
    'old':'262a10d6280eb6e79eec24e6170541a7b0f58e72',
    'v3':'36aab085bcfe8219997e1bc21e3ad8fe55f266b0',
    'v4':'edde403369ed87a90d497c307379e99881d5faab',
    'baseline':'19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5',
}


def read(p):return json.loads(p.read_text())
def digest_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(task,coefficient):
    baseline=task=='v4' and coefficient==.01
    if baseline:
        root=BASE/'artifacts/antmaze_v4_noveld001_reference'
        folder=root/'runs/v4-optiq-s0';checkpoint=root/'checkpoint-100k'
        proof=read(root/'baseline-100k-verification.json');sha=SOURCES['baseline']
    else:
        key='old' if task=='v3' and coefficient<=10 else task
        root=BASE/'artifacts'/dict(old='antmaze_noveld_strength_100k',v3='antmaze_noveld_strength_high_100k',v4='antmaze_noveld_strength_v4_high_100k')[key]
        job=f'{task}-optiq-c'+format(coefficient,'g').replace('.','p')+'-s0'
        folder=root/'runs'/job;checkpoint=folder/'resume/step_0000100000'
        proof=read(folder/'archive-verification.json');sha=SOURCES[key]
    assert proof['passed']
    c=read(folder/'config.json');manifest=read(checkpoint/'manifest.json')
    assert c['task']==task and c['seed']==0 and c['intrinsic']['coefficient']==coefficient
    assert c['source_commit']==manifest['source_commit']==sha and manifest['step']==100000
    assert manifest['updates']==95000
    for filename,spec in manifest['files'].items():
        assert digest_file(checkpoint/filename)==spec['sha256']
    s=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
    assert state_digest(s)==manifest['state_digest']
    with np.load(checkpoint/'replay.npz') as z:data={k:z[k] for k in z.files}
    assert state_digest(data)==manifest['replay_digest']
    xy=data['next_observations'][:,:2];assert len(xy)==100000
    assert s['replay']['position']==100000 and s['replay']['size']==100000
    walls=np.asarray(c['environment']['walls']);low=walls.min(0)-2;high=walls.max(0)+2
    counts=np.asarray(s['coverage']['counts']);assert counts.sum()+s['coverage']['outside']==100000
    index=np.floor((xy-low)/.5).astype(int)
    valid=(index>=0).all(1)&(index<counts.shape).all(1)
    recreated=np.zeros_like(counts);np.add.at(recreated,tuple(index[valid].T),1)
    # Replay float32 xy can round a boundary; retain exact simulator coverage
    # for headline metrics and expose any reconstruction discrepancy.
    difference=int(np.abs(recreated-counts).sum())
    assert difference<=20,('Unexpected occupancy mismatch',task,coefficient,difference)
    first=np.full(counts.shape,100001,dtype=int)
    np.minimum.at(first,tuple(index[valid].T),np.flatnonzero(valid)+1)
    timeline=np.arange(0,100001,1000)
    curve=np.array([(first<=step).sum() for step in timeline])
    goals=np.asarray(c['environment']['goals'])
    distance=np.linalg.norm(xy[:,None,:]-goals[None,:,:],axis=-1)
    np.testing.assert_allclose(data['rewards'],-distance.min(1),rtol=2e-6,atol=1e-5)
    mass=counts[counts>0]/counts.sum()
    entropy=float(-(mass*np.log(mass)).sum())
    left,right=(xy[:,0]<-2)&(xy[:,1]>2),(xy[:,0]>2)&(xy[:,1]<-2)
    if task=='v4':left,right=(xy[:,0]<-2)&(xy[:,1]>2),(xy[:,0]<-2)&(xy[:,1]<-2)
    regions=[left,right]
    def region_summary(mask):
        ix=index[mask&valid]
        return dict(step_fraction=float(mask.mean()),visited_bins=int(len(np.unique(ix,axis=0))),
                    min_x=float(xy[mask,0].min()) if mask.any() else None)
    terminals=data['dones'].astype(bool)
    terminal_goals=distance.argmin(1)
    row=dict(task=task,coefficient=coefficient,training_seed=0,steps=100000,updates=95000,
        visited_bins=int((counts>0).sum()),visited_area_grid_estimate_m2=float((counts>0).sum()*.25),
        bins_with_at_least10_visits=int((counts>=10).sum()),effective_bins=float(np.exp(entropy)),
        top10_bins_visit_fraction=float(np.sort(counts.ravel())[-10:].sum()/counts.sum()),
        region_definitions=['x < -2 and y > 2','x > 2 and y < -2'] if task=='v3' else ['x < -2 and y > 2','x < -2 and y < -2'],
        goal_direction_regions=[region_summary(mask) for mask in regions],
        min_goal_distances_m=distance.min(0).tolist(),training_successes_by_goal=[int((terminals&(terminal_goals==g)).sum()) for g in (0,1)],
        first_goal_steps=[int(np.flatnonzero(terminals&(terminal_goals==g))[0]+1) if (terminals&(terminal_goals==g)).any() else None for g in (0,1)],
        source_commit=sha,config_path=str(folder/'config.json'),checkpoint_path=str(checkpoint),
        checkpoint_files_sha256={k:v['sha256'] for k,v in manifest['files'].items()},
        replay_boundary_count_difference=difference,coverage_curve_steps=timeline.tolist(),coverage_curve_bins=curve.tolist(),
        evaluation=None,actual_bonus_last25k=None)
    if not baseline:
        result=read(folder/'result.json')
        row['evaluation']={label:result['summaries'][label] for label in ('native-fixed','policy-fixed')}
        logs=[]
        for line in (root/'logs'/f'training-{folder.name}.log').read_text().splitlines():
            if not line.startswith('{'):continue
            try:d=json.loads(line)
            except json.JSONDecodeError:continue
            if d.get('env_steps',0)>75000 and 'intrinsic_mean' in d:logs.append(d)
        row['actual_bonus_last25k']={k:float(np.mean([d[k] for d in logs])) for k in ('intrinsic_mean','env_reward_mean')}
    return dict(row=row,config=c,counts=counts,low=low,high=high)


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def draw_coverage(task,records):
    nrows=2 if len(records)>3 else 1
    fig,axes=plt.subplots(nrows,3,figsize=(13,5.2 if nrows==1 else 4.6*nrows),squeeze=False)
    maximum=max(r['counts'].max() for r in records)
    for ax,r in zip(axes.ravel(),records):
        row=r['row'];maze(ax,r['config'])
        values=r['counts'].astype(float);values[values==0]=np.nan
        im=ax.imshow(values.T,origin='lower',extent=[r['low'][0],r['high'][0],r['low'][1],r['high'][1]],
            cmap='YlOrRd',norm=LogNorm(1,maximum),interpolation='nearest',zorder=1)
        ax.set_title(f"NovelD {row['coefficient']:g} | {row['visited_bins']} bins\nEffective bins {row['effective_bins']:.0f} | top-10 visits {100*row['top10_bins_visit_fraction']:.1f}%",fontsize=10)
    fig.suptitle(f'AntMaze {task} | TRAINING exploration | 100k interactions | seed 0',fontsize=14,y=.99)
    fig.subplots_adjust(left=.02,right=.91,bottom=.10,top=.80 if nrows==1 else .92,wspace=.12,hspace=.32)
    fig.colorbar(im,cax=fig.add_axes([.935,.22,.012,.48]),label='Training visits (same scale)')
    fig.text(.5,.018,'Each bin = 0.5 m x 0.5 m. White = no visits. This is training coverage, not final-policy rollout diversity.',ha='center',fontsize=9)
    save(fig,task+'-training-coverage')


def main():
    records={task:[load(task,c) for c in coefficients] for task,coefficients in (('v3',(.1,1,5,10,50,100)),('v4',(.01,50,100)))}
    for task,rs in records.items():draw_coverage(task,rs)
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for ax,(task,rs) in zip(axes,records.items()):
        for r in rs:
            d=r['row'];ax.plot(np.asarray(d['coverage_curve_steps'])/1000,d['coverage_curve_bins'],label=f"NovelD {d['coefficient']:g}")
        ax.set(title=f'AntMaze {task}',xlabel='Environment interactions (k)',ylabel='Cumulative visited bins (0.5 m grid)',xlim=(0,100),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Exploration growth | same 100k budget | one training seed per coefficient')
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'cumulative-coverage')
    rows=[r['row'] for rs in records.values() for r in rs]
    report=dict(primary='training exploration',grid_bin_size_m=.5,training_seed_count=1,steps=100000,
        reporting_source_sha256=digest_file(Path(__file__)),results=rows)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# NovelD 탐색 범위 비교: 동일100k, OptiQ seed0','',
        '이번 단일 seed의 동일100k 비교에서는 NovelD50이 v3와v4 모두 가장 넓은 방문 영역을 만들었다. 100으로 더 높여도 범위가 더 커지지는 않았다.',
        'v3: 50의1277칸은 이전10의901칸보다41.7% 넓다. G2 최근접 거리도7.92m에서2.08m로 줄었다. 100은1131칸, G2 최근접8.25m로50보다 한쪽 목표 방향 탐색이 약했다.',
        'v4: .01의262칸에서50은511칸(1.95배),100은431칸(1.65배)으로 확대됐다. 50은 아래쪽과 중앙 통로로 더 멀리 들어갔고 G2에3.75m까지 접근했다(.01은8.91m,100은8.06m).',
        'v4의 방문분포 유효칸수는50과100이 모두약179로 비슷하다. 따라서50이 범위는넓지만 모든 방문편중지표까지 우월한 것은 아니다. 특히50도 가장많이머문10칸에23.4%가 남아있다.',
        '성공을기준으로선정하지않았다. 보조로v3-100은학습중G1에2회도달했고v3-50의최종full-policy평가는6/100성공했다. 나머지새실험의최종native/full-policy성공은0이며v3-50 native도0이다.',
        '',
        '주 기준은 학습중 탐색 범위·양쪽방향 탐색·방문편중이다. 성공은보조. 학습frozen소스는변경하지 않았다.',
        '각 .5m×.5m격자 방문을 센다. 면적환산은격자근사이며 로봇의 정확한 접근가능면적 비율이 아니다.',
        'Effective bins=exp(방문분포 엔트로피): 방문이 여러칸에 고르게 퍼질수록 증가한다. Top10은 가장많이머문10칸의시간비율이다.',
        '각 방향영역은 start주변2m를제외한좌표영역이며 성공경로/통로의종류를자동분류한지표가아니다.',
        '', '| 환경 | 계수 | 방문칸 | ≥10회방문칸 | 유효방문칸 | 상위10칸 시간비율 | G1/G2 최근접(m) | 학습성공G1/G2(보조) |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for d in rows:
        dist=d['min_goal_distances_m'];succ=d['training_successes_by_goal']
        lines.append(f"| {d['task']} | {d['coefficient']:g} | {d['visited_bins']} | {d['bins_with_at_least10_visits']} | {d['effective_bins']:.1f} | {100*d['top10_bins_visit_fraction']:.1f}% | {dist[0]:.2f}/{dist[1]:.2f} | {succ[0]}/{succ[1]} |")
    lines+=['','v4 .01은1M실험의100k보존checkpoint를사용했다. 최종1Mcoverage를섞지않았다.',
        '누적곡선은chronological replay float32좌표로재구성하고최종수치는고정밀simulator coverage를사용했다. 격자경계차이는results.json에기록했다.',
        '새4개와이전4개는전체checkpoint/replay/최종600rollout/run검증을통과했다. v4 .01은100k checkpoint SHA256/full-state/replay100k검증을통과했다.',
        '한trainingseed와100k범위의비교다. 탐색확대가향후성공향상을보장하지않으며,학습중다양한상태방문과최종정책의다양한경로는구분한다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ('task','coefficient','visited_bins','bins_with_at_least10_visits','effective_bins','top10_bins_visit_fraction','min_goal_distances_m','goal_direction_regions','training_successes_by_goal')} for r in rows],indent=2))


if __name__=='__main__':main()
