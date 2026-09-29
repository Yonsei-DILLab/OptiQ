"""Verify new checkpoint rollouts and show every random-start trajectory."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('route_audit', REPO/'artifacts/antmaze_dense_multimodality_audit_20260924/analyze.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
NAMES = ['legacy-v1', 'official-v1', 'legacy-v2', 'legacy-v3', 'legacy-v4']


def load(p):
    return json.loads(p.read_text())


def interval(k, n):
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def color(label):
    return audit.FAIL if label=='failure' else audit.NEG if label=='G1/lower' else audit.POS


def main():
    results, raw = {}, {}
    for name in NAMES:
        run = ROOT/name
        assert load(run/'result.json')['completed']
        proof = load(run/'verification.json')
        assert proof['passed'] and proof['checkpoint_unchanged'] and proof['paired_random_initial_states']
        p = load(run/'provenance.json')
        task = p['task']
        goals = (p['training_config']['environment']['goals'] if p['family']=='legacy' else audit.GOALS[task])
        results[name] = dict(provenance=p, modes={})
        for mode in ['policy','native']:
            path = run/f'{mode}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==proof['npz_sha256'][path.name]
            with np.load(path) as z:
                d = {k:z[k].copy() for k in z.files}
            n = len(d['lengths'])
            assert n==p['episodes_per_mode']==(1000 if task=='v1' else 500)
            np.testing.assert_allclose(d['initial_full_state'][:,:2],d['initial_positions'],atol=1e-12)
            assert len(np.unique(d['initial_positions'],axis=0))==n
            assert np.max(np.abs(d['initial_positions']))<2
            routes=[]
            for line,length,goal,ret,actions in zip(d['xy'],d['lengths'],d['goal_ids'],d['returns'],d['actions']):
                xy=line[:length+1]
                assert np.isfinite(xy).all() and np.isnan(line[length+1:]).all()
                assert np.isfinite(actions[:length]).all() and np.isnan(actions[length:]).all()
                assert np.max(np.abs(actions[:length]))<=1.000001
                dist=np.linalg.norm(xy[:,None,:]-np.asarray(goals)[None,:,:],axis=-1)
                assert bool(goal)==bool(dist.min()<=.50002)
                np.testing.assert_allclose(-dist[1:].min(axis=-1).sum(),ret,rtol=2e-6,atol=.004)
                routes.append(audit.route(task,xy,int(goal),goals))
            counts=Counter(routes)
            success=n-counts['failure']
            groups={}
            for label,mask in [('start_y_positive',d['initial_positions'][:,1]>=0),
                               ('start_y_negative',d['initial_positions'][:,1]<0)]:
                groups[label]=dict(episodes=int(mask.sum()),routes=dict(Counter(np.array(routes)[mask])))
            m=dict(episodes=n,successes=success,failures=counts['failure'],routes=dict(counts),
                   success_wilson95=interval(success,n),start_groups=groups,
                   route_wilson95={k:interval(v,n) for k,v in counts.items()},
                   mean_return=float(d['returns'].mean()))
            d['route_labels']=routes
            results[name]['modes'][mode]=m
            raw[(name,mode)]=d
        np.testing.assert_array_equal(raw[(name,'policy')]['initial_full_state'],raw[(name,'native')]['initial_full_state'])
    assert sum(m['episodes'] for r in results.values() for m in r['modes'].values())==7000
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(13.5,10.0))
    fig.subplots_adjust(left=.055,right=.985,top=.82,bottom=.17,hspace=.70,wspace=.26)
    for row,name in enumerate(['legacy-v1','official-v1']):
        for col,mode in [(0,'policy'),(2,'native')]:
            d,m=raw[(name,mode)],results[name]['modes'][mode]
            audit.draw(axes[row,col],'v1',d,m)
            axes[row,col].collections[-1].set_alpha(.15)
            axes[row,col].collections[-1].set_sizes([2])
            title='Direct policy: z + sigma' if mode=='policy' else 'Mu-only: random z, no sigma'
            axes[row,col].set_title(title+'\n'+f'+y {m["routes"].get("G1/upper",0)} | -y {m["routes"].get("G1/lower",0)} | fail {m["failures"]}',fontsize=10)
        ax=axes[row,1];d=raw[(name,'policy')]
        for label in ['failure','G1/lower','G1/upper']:
            mask=np.array(d['route_labels'])==label
            ax.scatter(d['initial_positions'][mask,0],d['initial_positions'][mask,1],s=10,
                       color=color(label),alpha=.6,edgecolors='none')
        ax.set(xlim=(-2.1,2.1),ylim=(-2.1,2.1),xlabel='Initial x (m)',ylabel='Initial y (m)')
        ax.set_aspect('equal');ax.axhline(0,color='#999999',lw=.7);ax.axvline(0,color='#999999',lw=.7)
        ax.set_title('Which route from each RANDOM start?\nColor = direct-policy outcome',fontsize=10)
    fig.suptitle('Fresh checkpoint evaluation | 1,000 random starts per policy and mode\nOptiQ AntMaze v1: successful routes and initial-position dependence',fontsize=15,y=.985)
    fig.text(.5,.88,'Earlier custom port | dense + NovelD 0.01 | 1M checkpoint',ha='center',fontsize=12,weight='bold')
    fig.text(.5,.495,'Official environment | dense + NovelD OFF | 3.008M checkpoint',ha='center',fontsize=12,weight='bold')
    handles=[Line2D([0],[0],color=audit.POS,lw=2,label='Successful +y route'),
             Line2D([0],[0],color=audit.NEG,lw=2,label='Successful -y route'),
             Line2D([0],[0],color=audit.FAIL,lw=2,label='Failure')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False)
    fig.text(.5,.023,'Every episode samples xy uniformly in [-2,2] x [-2,2]; paired starts across modes. All failures shown.\n'
             'One training seed per checkpoint. Original environment and policy preserved; rows are not a controlled NovelD ablation.',ha='center',fontsize=9)
    fig.savefig(ROOT/'v1_random_1000.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,6.2))
    fig.subplots_adjust(left=.055,right=.99,top=.77,bottom=.20,wspace=.22)
    for ax,task in zip(axes,['v2','v3','v4']):
        name='legacy-'+task;d,m=raw[(name,'policy')],results[name]['modes']['policy']
        audit.draw(ax,task,d,m)
        ax.collections[-1].set_alpha(.15)
        ax.collections[-1].set_sizes([3])
        route_count=len([k for k in m['routes'] if k!='failure' and 'unclassified' not in k])
        ax.set_title(f'{task.upper()} | success {m["successes"]}/500\nObserved successful corridor routes: {route_count}',fontsize=12)
    fig.suptitle('Fresh RANDOM-start evaluation | legacy dense + NovelD 0.01, final 1M\n500 direct-policy rollouts per maze; one training seed',fontsize=15,y=.98)
    fig.text(.5,.052,'xy uniform[-2,2] is an explicit evaluation override for v2-v4 (training reset was fixed).\n'
             'Blue: success; gray: failure; red: failed endpoint. Fresh z + conditional sigma; no external DACER noise.',fontsize=10,ha='center')
    fig.savefig(ROOT/'v234_random_500.png',dpi=170);plt.close(fig)
    (ROOT/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    old=results['legacy-v1']['modes']['policy'];off=results['official-v1']['modes']['policy']
    lines=['# Dense OptiQ 체크포인트: 새로운 랜덤 시작 재평가','',
           '기존 저장 궤적 재집계가 아니라, 저장된 학습 완료 체크포인트를 복원해 새로 7,000 episode를 rollout했다. 학습은 수행하지 않았다.',
           f'이전 포트 v1은 위쪽 {old["routes"].get("G1/upper",0)}, 아래쪽 {old["routes"].get("G1/lower",0)}, 실패 {old["failures"]}로 두 성공 경로를 고르게 사용했다.',
           f'공식 환경 NovelD OFF v1도 위쪽 {off["routes"].get("G1/upper",0)}회가 관찰되어 두 경로가 존재한다. 다만 아래쪽 {off["routes"].get("G1/lower",0)}회로 성공의 {100*off["routes"].get("G1/lower",0)/off["successes"]:.1f}%가 한쪽에 편중됐다.',
           '매 episode마다 xy를 [-2,2]×[-2,2]에서 독립 균등 추출하고 원래 초기 자세·속도를 유지했다. policy와 native에 같은 시작 상태 묶음을 사용했다. 단일 학습 seed0. 고정 시작 결과는 이 보고서에 섞지 않았다.',
           'policy=fresh random z+conditional sigma. native=fresh random z mu-only. 외부 DACER 행동잡음 및 NovelD 평가보상 없음.', '',
           '|모델|평가 모드|성공/평가수|성공 경로 및 실패|', '|---|---|---:|---|']
    for name in NAMES:
        for mode,m in results[name]['modes'].items():
            lines.append(f'|{name}|{mode}|{m["successes"]}/{m["episodes"]}|'+', '.join(f'{k}: {v}' for k,v in m['routes'].items())+'|')
    lines+=['','## 시작 위치와 선택 경로','',
            '같은 랜덤 시작 분포에서도 시작 좌표에 따라 경로 선택이 달라질 수 있다. 아래는 v1 direct-policy에서 초기 y의 부호별 결과다. 위치별 조건부 선택과 전체 분포의 경로 다양성을 구분한다.','',
            '|모델|시작 영역|평가수|+y 성공|−y 성공|실패|','|---|---|---:|---:|---:|---:|']
    for name in ['legacy-v1','official-v1']:
        for label,g in results[name]['modes']['policy']['start_groups'].items():
            c=g['routes'];lines.append(f'|{name}|{label}|{g["episodes"]}|{c.get("G1/upper",0)}|{c.get("G1/lower",0)}|{c.get("failure",0)}|')
    lines+=['','## 검증과 해석 범위','',
            '- 체크포인트 SHA256, 복원된 actor/critic/optimizer 값 일치 및 평가 후 불변을 확인했다.',
            '- 저장 NPZ SHA256, 모든 action/xy 유한성 및 action 범위, 실제 goal 반경 도달, dense 거리 보상 합계를 로컬에서 다시 검증했다.',
            '- 평가 시작점은 모두 서로 다르며 두 평가 모드에서 전체 qpos/qvel이 정확히 같다.',
            '- 이전 v2-v4의 학습 시작은 고정이었다. 여기서는 사용자 요청에 따라 평가만 랜덤 시작으로 변경했다. 따라서 초기상태 분포 이동이 포함된다.',
            '- 공식 환경 dense OFF v2-v4는 당시 final-only 저장 전 중단되어 모델이 남아 있지 않았다. 재평가한 v2-v4는 이전 포트 + NovelD0.01 완료 모델이다.',
            '- 각기 다른 frozen 환경 구현과 학습조건을 그대로 사용했다. NovelD 단독 인과효과나 여러 학습 seed의 재현성을 주장하지 않는다.',
            '- Wilson 95% 구간은 저장된 단일 정책의 rollout 변동에 관한 구간이며 학습 seed 간 불확실성을 나타내지 않는다.',
            '- 랜덤 초기상태 분포에서의 여러 경로는 유효한 행동 다양성이다. 이것만으로 같은 state의 action density가 다봉이라는 주장을 하지는 않는다.',
            '- CPU-only 네트워크 추론과 MuJoCo rollout이다. 학습 GPU/체크포인트/현재16개 캠페인 설정은 변경하지 않았다.',
            '- initial_xy seed sequence 및 정책 난수는 provenance.json에 기록했다. 전체 raw actions/xy/initial states를 보관한다.',
            '', '## 파일','',
            '- `v1_random_1000.png`: 랜덤 시작 궤적·시작 위치별 경로·mu-only 비교',
            '- `v234_random_500.png`: v2-v4 새 랜덤 시작 direct-policy 궤적',
            '- `results.json`: 전체 지표, 초기 y별 집계, 성공 및 경로 비율 Wilson 95% 구간',
            '- 각 모델 폴더: `policy.npz`, `native.npz`, `provenance.json`, `verification.json`',
            '- 초기 공식 모델 smoke의 직렬화 바이트 비교는 map 순서 때문에 실패했다. 이름별 모든 값·dtype·shape 일치 검사로 고친 후 별도 smoke를 통과했으며 실패 자료를 보존했다. 학습 모델 문제는 아니었다.']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{m:{'success':v['successes'],'episodes':v['episodes'],'routes':v['routes'],'start_groups':v['start_groups']} for m,v in r['modes'].items()} for k,r in results.items()},indent=2))


if __name__=='__main__':
    main()
