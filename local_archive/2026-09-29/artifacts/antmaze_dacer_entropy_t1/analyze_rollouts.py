"""Validate and classify saved rollout routes; do not pool separate policies."""
import importlib.util
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = '4a33d13f0a8de3be508537fd38879ac8ef8018c0'
spec = importlib.util.spec_from_file_location('route_audit',
    REPO/'artifacts/antmaze_dense_multimodality_audit_20260924/analyze.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
load = lambda p: json.loads(p.read_text())


def read_mode(folder, task):
    summary = load(folder/'summary.json')
    fixed = folder.name.endswith('-fixed')
    d, m = audit.read_rollout(folder/'rollouts.npz', task, fixed=fixed)
    n = m['episodes']
    assert n == summary['episodes'] and m['successes'] == round(summary['success_rate']*n)
    assert int(d['env_steps']) == summary['step']
    assert str(d['mode']) == summary['mode'] and bool(d['fixed']) == summary['fixed']
    assert n in (40, 100), n
    starts = d['initial_full_state'][:, :2]
    np.testing.assert_allclose(d['xy'][:, 0], starts, atol=1e-6)
    assert np.max(np.abs(starts)) <= 2.00001
    if not fixed:
        assert len(np.unique(starts, axis=0)) == n
    m.update(step=summary['step'], mode=folder.name, mean_return=float(d['returns'].mean()))
    if task == 'v1':
        m['initial_y_by_route'] = {
            route: dict(count=sum(x==route for x in d['route_labels']),
                        positive=int(sum(x==route and y>0 for x,y in zip(d['route_labels'],starts[:,1]))),
                        negative=int(sum(x==route and y<0 for x,y in zip(d['route_labels'],starts[:,1]))))
            for route in sorted(set(d['route_labels']))}
    if task == 'v3':
        branches=[]
        for path, length in zip(d['xy'], d['lengths']):
            line=path[:length+1]
            left=bool((line[:,0] < -8).any())
            right=bool((line[:,0] > 8).any())
            branches.append('both' if left and right else 'left' if left else 'right' if right else 'neither')
        m['passage_visits_including_failures']=dict(Counter(branches))
    return d, m


def main():
    snapshot = Path((ROOT/'latest-report-path.txt').read_text().strip())
    runs, arrays, states = [], {}, {}
    for host in ('vast-heechan-180', 'vast-heechan-199'):
        base = snapshot/host
        state = load(base/'status.json')
        states[host] = state
        for run in sorted((base/'runs').glob('*')):
            if not (run/'config.json').is_file():continue
            cfg = load(run/'config.json')
            assert cfg['source_commit'] == SOURCE and cfg['method'] == 'optiq'
            assert cfg['seed'] == 0 and cfg['reward_profile'] == 'dense'
            assert not cfg['noveld_enabled'] and cfg['eval_starts'] == 'random'
            assert cfg['temperature'] == 1.0
            policies = sorted((run/'evaluations').glob('*/policy-natural/rollouts.npz'))
            if not policies:continue
            history = []
            for raw in policies:
                d, m = read_mode(raw.parent, cfg['task'])
                history.append(m)
            latest = history[-1]
            modes = {'policy-natural': latest}
            arrays[run.name, 'policy-natural'] = d
            for folder in sorted(policies[-1].parent.parent.iterdir()):
                if folder.name == 'policy-natural' or not (folder/'rollouts.npz').is_file():continue
                dm, mm = read_mode(folder, cfg['task'])
                modes[folder.name] = mm
                arrays[run.name, folder.name] = dm
            result = load(run/'result.json') if (run/'result.json').is_file() else {}
            progress = load(run/'progress.json') if (run/'progress.json').is_file() else {}
            if result.get('completed'):
                assert result['steps'] == cfg['steps'] and result['rnd_updates'] == 0
            status = ('completed' if result.get('completed') else
                      'final evaluation/checkpoint' if progress.get('step',0) >= cfg['steps'] else 'training')
            runs.append(dict(id=run.name, host=host, task=cfg['task'],
                             target_per_dim=cfg['dacer_target_entropy_per_dim'],
                             source=cfg['source_commit'], status=status,
                             progress_step=progress.get('step'), budget=cfg['steps'],
                             modes=modes, history=history))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'white'})
    fig, axes = plt.subplots(3,3,figsize=(13.8,13.8))
    fig.subplots_adjust(top=.9,bottom=.10,left=.055,right=.98,wspace=.28,hspace=.4)
    for i,target in enumerate((-1.,-.8,-.5)):
        for j,task in enumerate(('v1','v3','v4')):
            ax=axes[i,j]
            row=next((r for r in runs if r['task']==task and r['target_per_dim']==target),None)
            if row is None:
                audit.decorate(ax,task)
                ax.text(.5,.5,'No saved evaluation yet',ha='center',transform=ax.transAxes)
                ax.set_title(f'{task.upper()} | target/dim {target:g}')
                continue
            m=row['modes']['policy-natural']
            audit.draw(ax,task,arrays[row['id'],'policy-natural'],m)
            ax.set_title(f'{task.upper()} | target/dim {target:g} | {m["step"]/1e6:.3f}M\n'
                         f'Success {m["successes"]}/{m["episodes"]} | observed routes {len(m["successful_routes"])}',fontsize=10)
    fig.suptitle('DACER target-entropy sweep | latest saved direct-policy trajectories\n'
                 'OptiQ T=1 | dense reward, NovelD OFF | training seed 0',fontsize=16,y=.97)
    fig.legend(handles=[Line2D([0],[0],color=audit.POS,lw=2,label='Success'),
                        Line2D([0],[0],color=audit.NEG,lw=2,label='v1 lower-route success'),
                        Line2D([0],[0],color=audit.FAIL,lw=2,label='Failure; red endpoint')],
               loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    fig.text(.5,.012,'40 interim / 100 final random-start episodes; all failures included. Each panel is a separate policy.\n'
             'Random z + conditional sigma; no external DACER noise. Different training steps are shown, not a matched-budget ranking.\n'
             'Multiple routes from different starts do not establish same-state route diversity.',ha='center',fontsize=9)
    fig.savefig(snapshot/'latest_trajectories.png',dpi=155)
    plt.close(fig)

    final_v1=[r for r in runs if r['task']=='v1' and 'policy-fixed' in r['modes']]
    if final_v1:
        fig,axes=plt.subplots(len(final_v1),2,figsize=(11,4.5*len(final_v1)),squeeze=False)
        for i,row in enumerate(final_v1):
            for j,mode in enumerate(('policy-natural','policy-fixed')):
                m=row['modes'][mode];ax=axes[i,j]
                audit.draw(ax,'v1',arrays[row['id'],mode],m)
                ax.set_title(f'target/dim {row["target_per_dim"]:g} | '+('Random starts' if j==0 else 'Identical full state')+
                             f'\nUpper {m["routes"].get("G1/upper",0)} | lower {m["routes"].get("G1/lower",0)} | fail {m["failures"]}')
        fig.suptitle('v1 | final direct-policy rollouts | 100 episodes per panel',fontsize=15)
        fig.tight_layout(rect=(0,.045,1,.95))
        fig.text(.5,.015,'Same trained policy within each row. Random z + conditional sigma; no external DACER noise.',ha='center',fontsize=10)
        fig.savefig(snapshot/'v1_same_start.png',dpi=160);plt.close(fig)

    output=dict(collection=load(snapshot/'collection-verification.json'), source=SOURCE,
                single_training_seed=0, raw_validation_passed=True,
                status_counts={h:{k:len(s.get(k,[])) for k in ('completed','running','pending','failed')} for h,s in states.items()},
                input_sha256=audit.INPUTS,runs=runs)
    (snapshot/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# DACER entropy sweep 중간 경로 분석','',
           '학습 seed 0, OptiQ T=1, dense reward + NovelD OFF. 저장된 원시 direct-policy rollout만 재집계했다. 외부 DACER 잡음은 평가에 없다.','',
           '|미로|목표/차원|학습 진행|평가 step|성공/평가수|성공 경로|동일 상태 평가|',
           '|---|---:|---:|---:|---:|---|---|']
    for row in sorted(runs,key=lambda r:(r['task'],r['target_per_dim'])):
        m=row['modes']['policy-natural'];fixed=row['modes'].get('policy-fixed')
        lines.append(f'|{row["task"]}|{row["target_per_dim"]:g}|{row["progress_step"]}|{m["step"]}|{m["successes"]}/{m["episodes"]}|'
                     +json.dumps(m['successful_routes'])+'|'+(json.dumps(fixed['routes']) if fixed else '최종 평가 전')+'|')
    lines+=['','실패 궤적을 제외하지 않고 그림에 함께 표시했다. 목표 도달과 통로 교차를 원시 좌표로 검증했으며 dense return 거리합, 초기 상태, 유한성, padding, SHA256도 확인했다.',
            '랜덤 시작점별 다른 경로는 동일 상태 정책 다양성과 구분해야 한다. 서로 다른 H_target 정책의 경로를 합쳐 한 정책의 다양성으로 세지 않는다.',
            '중간 평가는 40회, 최종은 100회이며 서로 학습 진행도가 달라 성능 순위로 해석하지 않는다. 성공이 없는 정책에 대해 성공 경로가 붕괴했다고 단정하지 않는다.',
            '중간 모델/학습 상태를 변경하거나 새 학습·rollout을 실행하지 않았다.']
    (snapshot/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(snapshot=str(snapshot),status_counts=output['status_counts'],runs=[
        {k:r[k] for k in ('id','task','target_per_dim','status','progress_step','budget','modes')} for r in runs]),indent=2))


if __name__ == '__main__':main()
