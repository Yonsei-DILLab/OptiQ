"""Collect completed, paired physical diagnostics; never launch or change a policy."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

OUT = Path(__file__).resolve().parent
ROOT = '/home/heechan/optiq-experiments/antmaze-v4-wall-contact-diagnostic-20260925'
SOURCE = 'f21108190afa212a811b78ca222b2bf7ce086489'
TRAINING = '555bb7e3c0101ee939545cc35d4d0729291781ec'
WORKTREE = OUT.parents[1] / 'tmp/reward-progress-worktree'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect():
    code = r'''
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile,time
root=Path(ROOT)
result=json.loads((root/'evaluation/result.json').read_text())
assert result['completed'] and result['verification']['passed']
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
status=subprocess.run(ctl+['status',root.name],text=True,capture_output=True).stdout.strip()
assert 'EXITED' in status, status
metadata={'time':time.time(),'root':str(root),'status':status,'files':{},'read_only':True}
paths=[root/'registration.json']
paths+=sorted((root/'evaluation').glob('*.json'))
paths+=sorted((root/'evaluation').glob('*.npz'))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as archive:
 for p in paths:
  before=p.stat();data=p.read_bytes();after=p.stat()
  assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  name=str(p.relative_to(root))
  metadata['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
 data=json.dumps(metadata,indent=2).encode()
 item=tarfile.TarInfo('collection-metadata.json');item.size=len(data);archive.addfile(item,io.BytesIO(data))
'''
    command = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','vast-heechan-199','python3','-']
    answer = subprocess.run(command,input=('ROOT='+repr(ROOT)+'\n'+code).encode(),capture_output=True,timeout=180,check=True)
    with tarfile.open(fileobj=io.BytesIO(answer.stdout),mode='r:gz') as archive:
        members = {}
        for item in archive.getmembers():
            name=PurePosixPath(item.name)
            assert item.isfile() and not name.is_absolute() and '..' not in name.parts
            members[item.name]=archive.extractfile(item).read()
    metadata=json.loads(members['collection-metadata.json'])
    for name,record in metadata['files'].items():
        assert len(members[name])==record['bytes']
        assert hashlib.sha256(members[name]).hexdigest()==record['sha256']
    for name,data in members.items():
        p=OUT/'raw'/name;p.parent.mkdir(parents=True,exist_ok=True)
        temporary=p.with_suffix(p.suffix+'.download');temporary.write_bytes(data);temporary.replace(p)


def report():
    root=OUT/'raw/evaluation'
    result=json.loads((root/'result.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    assert result['training_source']==TRAINING and result['evaluation_source']==SOURCE
    assert result['completed'] and all(result['verification'][k] for k in (
        'passed','checkpoint_unchanged','model_optimizer_unchanged','evaluation_rng_restored',
        'paired_xy_actions_returns_exact','identical_initial_full_state','native_horizon_unchanged'))
    arrays={}
    for mode in ('plain','instrumented'):
        p=root/(mode+'.npz')
        assert digest(p)==result['conditions'][mode]['sha256']
        with np.load(p,allow_pickle=False) as z:arrays[mode]={k:z[k] for k in z.files}
    plain,data=arrays['plain'],arrays['instrumented']
    for k in plain:np.testing.assert_array_equal(plain[k],data[k])
    assert len(data['lengths'])==40
    np.testing.assert_array_equal(data['initial_full_state'],np.repeat(data['initial_full_state'][:1],40,axis=0))
    assert np.all(data['initial_full_state'][:,:2]==0)
    module_path=WORKTREE/'antmaze_experiments/progress_reward.py'
    expected=subprocess.check_output(['git','show',TRAINING+':antmaze_experiments/progress_reward.py'],cwd=WORKTREE)
    assert hashlib.sha256(expected).hexdigest()==digest(module_path)
    spec=importlib.util.spec_from_file_location('verified_progress_reward',module_path)
    reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)
    rows=[dict(row) for row in result['episode_diagnostics']]
    assert len(rows)==40
    errors=[]
    for i,n in enumerate(data['lengths'].astype(int)):
        assert 1<=n<=700
        for k in ('xy','actions','qpos','qvel','wall_count','wall_min_distance'):
            used=n+int(k=='xy');assert np.isfinite(data[k][i,:used]).all()
        np.testing.assert_allclose(data['qpos'][i,:n,:2],data['xy'][i,:n],atol=2e-6,rtol=1e-7)
        assert np.all(data['wall_count'][i,:n]>=0) and np.all(data['wall_min_distance'][i,:n]<=0)
        assert np.max(np.abs(data['actions'][i,:n]))<=1+1e-6
        profile=provenance['training_config']['reward_profile']
        expected_return=100*(reward.distance(data['xy'][i,0],'v4',profile)-reward.distance(data['xy'][i,n],'v4',profile))
        errors.append(abs(data['returns'][i]-expected_return))
        assert rows[i]['episode']==i and rows[i]['steps']==n
        np.testing.assert_allclose(rows[i]['last200_wall_contact_fraction'],np.mean(data['wall_count'][i,max(0,n-200):n]>0))
        rows[i]['last200_geodesic_progress']=float(
            reward.distance(data['xy'][i,max(0,n-200)],'v4',profile)
            -reward.distance(data['xy'][i,n],'v4',profile))
    assert max(errors)<.01
    metrics=('last200_wall_contact_fraction','last200_net_displacement','last200_path_length',
             'last200_planar_speed_rms','last200_mean_height','final_geodesic','last200_geodesic_progress')
    summary={}
    for route in sorted({r['route'] for r in rows}):
        selected=[r for r in rows if r['route']==route]
        summary[route]={'episodes':len(selected),'successes':sum(bool(r['goal']) for r in selected)}
        summary[route]['mean']={k:float(np.mean([r[k] for r in selected])) for k in metrics}
        summary[route]['median']={k:float(np.median([r[k] for r in selected])) for k in metrics}
        summary[route]['contact_over_half_and_displacement_under_1m']=sum(
            r['last200_wall_contact_fraction']>.5 and r['last200_net_displacement']<1 for r in selected)
    output=OUT/'report';output.mkdir(exist_ok=True)
    colors={'upper':'#2374ab','lower':'#df7b28','uncommitted':'#777777'}
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    walls,goals,bounds=reward.maze_geometry('v4')
    physical=[]
    geometry=result['geometry']
    for i in geometry['wall_ids']:
        center=np.asarray(geometry['geom_pos'][i][:2]);half=np.asarray(geometry['geom_size'][i][:2])
        physical.append(tuple(np.r_[center-half,center+half]))
    np.testing.assert_allclose(sorted(physical),sorted(map(tuple,walls)),rtol=0,atol=1e-9)
    routes=[r for r in ('upper','lower') if r in summary]
    for ax,route in zip(axes[0],routes):
        for x0,y0,x1,y1 in walls:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#e0e3e8',ec='#959ba5',lw=.5))
        for x,y in goals:
            ax.scatter(x,y,s=120,marker='*',c='#289148',zorder=5)
            ax.add_patch(Circle((x,y),.5,fill=False,color='#289148'))
        for i,row in enumerate(rows):
            if row['route']!=route:continue
            n=int(data['lengths'][i]);xy=data['xy'][i,:n+1]
            ax.plot(xy[:,0],xy[:,1],color=colors[route],alpha=.4,lw=.65)
            mask=data['wall_count'][i,:n]>0
            ax.scatter(xy[:n][mask,0],xy[:n][mask,1],s=1,c='#d53636',alpha=.2,rasterized=True)
            ax.scatter(*xy[-1],marker='x',s=18,c='#111111',lw=.8)
        ax.scatter(0,0,s=35,c='black',zorder=6)
        ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),aspect='equal',xlabel='x (m)',ylabel='y (m)',
               title=f"{route.title()}: {summary[route]['episodes']} episodes, {summary[route]['successes']} goals")
    for route in summary:
        selected=[r for r in rows if r['route']==route]
        axes[1,0].scatter([100*r['last200_wall_contact_fraction'] for r in selected],
                          [r['last200_net_displacement'] for r in selected],label=route,c=colors.get(route,'gray'),s=30,alpha=.8)
        axes[1,1].scatter([r['last200_mean_height'] for r in selected],
                          [r['final_geodesic'] for r in selected],label=route,c=colors.get(route,'gray'),s=30,alpha=.8)
    axes[1,0].set(xlabel='Last 200 steps: wall contact (%)',ylabel='Last 200 steps: net displacement (m)')
    axes[1,1].set(xlabel='Last 200 steps: mean torso height (m)',ylabel='Final geodesic distance (m)')
    for ax in axes[1]:ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('v4 frozen 258k policy: physical contact audit\n40 direct-policy episodes, identical full start, native 700-step horizon\nRed = actual wall contacts; black crosses = endpoints',fontsize=13)
    fig.savefig(output/'wall_contact_diagnostic.png',dpi=160);plt.close(fig)
    verified={'time_utc':datetime.now(timezone.utc).isoformat(),'training_source':TRAINING,'evaluation_source':SOURCE,
              'reporter_sha256':digest(__file__),'max_reward_error':float(max(errors)),'paired_raw_arrays_exact':True,
              'checkpoint_sha256':provenance['checkpoint_sha256'],'route_summary':summary,
              'derived_episode_diagnostics':rows,
              'limitations':'One seed, one frozen policy, 40 same-origin direct rollouts; contact is observational, not a causal intervention. Native 700 horizon unchanged; original 100-episode primary report retained.'}
    (output/'results.json').write_text(json.dumps(verified,indent=2)+'\n')
    lines=['# v4 저장 정책의 벽 접촉 진단','',
           '250k 학습 정책 하나로 같은 전체 초기 상태에서 직접 정책을 40회 평가했습니다. 원래 700-step 종료 조건과 정책을 유지했습니다. 접촉 기록 전후의 행동·궤적·보상은 정확히 일치합니다.','',
           '| 경로 | 횟수 | 성공 | 마지막 200 step 벽 접촉 | 순이동 | 몸통 높이 | 마지막 geodesic 거리 | 마지막 200 step 거리 감소 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for route,s in summary.items():
        m=s['mean'];lines.append(f"| {route} | {s['episodes']} | {s['successes']} | {m['last200_wall_contact_fraction']:.1%} | {m['last200_net_displacement']:.2f}m | {m['last200_mean_height']:.2f}m | {m['final_geodesic']:.2f}m | {m['last200_geodesic_progress']:.2f}m |")
    lines+=['','표의 연속값은 경로 내 episode 평균입니다. 이번 자료에서 마지막 200 step의 이동 경로 길이는 약 9m이지만 순이동은 0.8~1.2m입니다. 몸통 평균 높이는 0.57~0.58m이고 벽 접촉 비율은 약 6~7%입니다. 계속 벽에 접촉한 채 정지한다는 설명은 지지되지 않습니다. 목표까지 남은 geodesic 거리는 마지막 200 step 동안 위쪽에서 평균 0.36m, 아래쪽에서 0.86m 줄었습니다. 따라서 완전히 진전이 없는 것이 아니라, 이동량에 비해 목표 방향의 진전이 느리며 원래 700-step 제한 내에는 도달하지 못한 것입니다.','',
            '빨간 점은 Ant와 벽의 실제 접촉 기록이며 바닥·몸체 자기 접촉은 제외했습니다. 각 action 직전, 마지막 physics substep의 접촉 cache를 읽었으므로 substep 사이의 모든 접촉이나 충격량을 측정한 것은 아닙니다. 접촉이 학습 실패에 전혀 영향을 주지 않는다는 뜻도 아닙니다. 한 정책·한 seed의 관찰 진단이며 해결책의 인과 효과는 아직 검증하지 않았습니다.','',
            '![벽 접촉 진단](wall_contact_diagnostic.png)','',
            f'학습 source: `{TRAINING}`. 평가 source: `{SOURCE}`. 보고 코드는 평가 완료 후 원자료에 적용했으며 SHA는 results.json에 보관합니다.']
    (output/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verified,ensure_ascii=False))


if __name__=='__main__':
    collect()
    report()
