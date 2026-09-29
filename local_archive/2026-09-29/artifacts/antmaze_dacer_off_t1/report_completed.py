"""Read-only reporting of verified saved policies, including incomplete paths."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('saved_routes', ROOT.parent/'antmaze_dacer_entropy_t1/analyze_rollouts.py')
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
audit = tools.audit
COLORS = dict(upper='#2563eb', lower='#d98513', left='#2563eb', right='#d98513', both='#953bc4', uncommitted='#77818c')
LABELS = dict(upper='Upper', lower='Lower', left='Left', right='Right', both='Both sides', uncommitted='No gate')


def family(task, line):
    if task in ('v1', 'v4'):
        y = audit.crossing(line, -4)
        return 'upper' if y is not None and y > 2 else 'lower' if y is not None and y < -2 else 'uncommitted'
    gate = 4 if task == 'v2' else 8
    left, right = (line[:, 0] < -gate).any(), (line[:, 0] > gate).any()
    return 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'


def detail(task, mode):
    data, metrics = tools.read_mode((REPO/mode['input']).parent, task)
    rows = []
    for i, (xy, length, goal) in enumerate(zip(data['xy'], data['lengths'], data['goals'])):
        line = xy[:int(length)+1]
        distances = np.linalg.norm(line[:, None, :] - np.asarray(audit.GOALS[task])[None, :, :], axis=-1)
        gates = {}
        for gate in (-12, -8, -4, 4, 8):
            y = audit.crossing(line, gate, side=-1 if gate < 0 else 1)
            if y is not None:
                gates[str(gate)] = float(y)
        rows.append(dict(episode=i, family=family(task, line), success=bool(goal), goal=int(goal),
                         length=int(length), start=line[0].tolist(), endpoint=line[-1].tolist(),
                         closest_goal_m=float(distances.min()), closest_each_goal_m=distances.min(0).tolist(),
                         last100_bbox_m=float(np.linalg.norm(np.ptp(line[-100:], axis=0))),
                         first_crossing_y=gates))
    counts = dict(Counter(r['family'] for r in rows))
    assert sum(counts.values()) == metrics['episodes']
    return data, dict(step=metrics['step'], episodes=metrics['episodes'], successes=metrics['successes'],
                     counts=counts, success_by_family=dict(Counter(r['family'] for r in rows if r['success'])),
                     failures_by_family=dict(Counter(r['family'] for r in rows if not r['success'])), rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot or Path((ROOT/'latest-report-path.txt').read_text().strip())
    source = json.loads((snapshot/'analysis.json').read_text())
    prior_path = REPO/'artifacts/antmaze_anneal_baselines/reports/20260924T050707Z/analysis.json'
    prior = json.loads(prior_path.read_text())
    assert source['raw_validation_passed'] and prior['raw_validation_passed']
    out = snapshot/'completed_report'
    out.mkdir(exist_ok=True)
    arrays, details, histories, comparisons = {}, {}, {}, {}
    for run in source['runs']:
        task = run['task']
        details[task] = {}
        for mode, metrics in run['modes'].items():
            arrays[task, mode], details[task][mode] = detail(task, metrics)
        histories[task] = []
        for metrics in run['history']:
            _, d = detail(task, metrics)
            histories[task].append({k:v for k,v in d.items() if k != 'rows'})
        on = prior['runs']['T=1'][task]
        off_cfg_path = (REPO/run['modes']['policy-natural']['input']).parents[3]/'config.json'
        on_cfg_path = (REPO/on['modes']['policy-natural']['input']).parents[3]/'config.json'
        off_cfg, on_cfg = json.loads(off_cfg_path.read_text()), json.loads(on_cfg_path.read_text())
        diff = {k:dict(on=on_cfg.get(k), off=off_cfg.get(k)) for k in set(on_cfg)|set(off_cfg) if on_cfg.get(k)!=off_cfg.get(k)}
        comparisons[task] = dict(config_differences=diff, on_final=on['modes']['policy-natural'],
                                 off_latest=run['modes']['policy-natural'], off_completed=run['status']=='completed')

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.8))
    fig.subplots_adjust(top=.86, bottom=.10, left=.065, right=.985, wspace=.24, hspace=.42)
    for ax, run in zip(axes.flat, source['runs']):
        task = run['task']; d = arrays[task, 'policy-natural']; info = details[task]['policy-natural']
        audit.decorate(ax, task)
        for r in info['rows']:
            line = d['xy'][r['episode'], :r['length']+1]; color = COLORS[r['family']]
            ax.plot(line[:, 0], line[:, 1], color=color, alpha=.33, lw=.75, zorder=2)
            ax.scatter(*line[0], color=color, marker='^', s=10, alpha=.55, zorder=4)
            ax.scatter(*line[-1], color='#bd2942' if not r['success'] else color,
                       marker='x' if not r['success'] else 'o', s=21 if not r['success'] else 10,
                       lw=.9 if not r['success'] else 0, alpha=.85, zorder=6)
        state = 'FINAL' if run['status']=='completed' else 'INTERIM'
        ax.set_title(f'{task.upper()} | {state} {info["step"]/1e6:.3f}M | success {info["successes"]}/{info["episodes"]}\n'
                     + ' / '.join(f'{LABELS[k]} {v}' for k,v in sorted(info['counts'].items())), fontsize=10)
        ax.legend(handles=[Line2D([0],[0],color=COLORS[k],lw=2,label=f'{LABELS[k]} ({v})') for k,v in sorted(info['counts'].items())],
                  fontsize=8, loc='upper right' if task=='v3' else 'lower right', framealpha=.94)
    fig.suptitle('OptiQ: DACER OFF, T=1 | latest saved trajectories\nEvery rollout included; colors show corridor choices, independent of success', y=.97, fontsize=15)
    fig.text(.5,.025, 'Dense reward, NovelD OFF, one training seed (0). Random starts; random z + conditional sigma.\nTriangles = starts; red crosses = unsuccessful endpoints; stars = goals. Final 100 / interim 40 episodes.', ha='center', fontsize=10)
    fig.savefig(out/'all_trajectories.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.7))
    fig.subplots_adjust(top=.85, bottom=.15, hspace=.42, wspace=.23)
    for ax, run in zip(axes.flat, source['runs']):
        task=run['task']; on=prior['runs']['T=1'][task]['history']; off=run['history']
        ax.plot([h['step']/1e6 for h in on], [100*h['success_rate'] for h in on], color='#788a9d', ls='--', lw=1.8, label='DACER ON (target/dim -0.9)')
        ax.plot([h['step']/1e6 for h in off], [100*h['successes']/h['episodes'] for h in off], color='#163cd2', lw=1.9, marker='o', ms=3, label='DACER OFF')
        ax.set(title=task.upper(), xlabel='Environment interactions (M)', ylabel='Success (%)', ylim=(-3,103))
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Saved evaluation histories | T=1, same environment budgets\n40 random-start episodes per checkpoint; 100 at final; no smoothing',y=.97,fontsize=14)
    fig.text(.5,.025, 'One training seed per condition. These are descriptive comparisons, not seed-averaged treatment effects.\nDACER OFF lines stop at their latest saved evaluation; DACER ON controls are completed.',ha='center',fontsize=9)
    fig.savefig(out/'learning_curves.png',dpi=170);plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(11.4,7.8))
    fig.subplots_adjust(top=.86,bottom=.15,hspace=.46,wspace=.25)
    for ax,run in zip(axes.flat,source['runs']):
        task=run['task']; rows=histories[task]; keys=sorted({k for r in rows for k in r['counts']})
        for k in keys:
            ax.plot([r['step']/1e6 for r in rows], [100*r['counts'].get(k,0)/r['episodes'] for r in rows],
                    color=COLORS[k], marker='o',ms=3,lw=1.8,label=LABELS[k])
        ax.set(title=task.upper(),xlabel='Environment interactions (M)',ylabel='All rollouts (%)',ylim=(-3,103))
        ax.grid(alpha=.18);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Corridor use over training | successes and failures together\nEach point is one saved policy; trajectories from different policies are not pooled',fontsize=14,y=.97)
    fig.text(.5,.025, 'No gate = did not cross the geometric corridor gate. Small oscillations within one corridor are not additional modes.\nGate definitions and individual trajectories are saved in analysis.json; one seed only.',ha='center',fontsize=9)
    fig.savefig(out/'corridor_history.png',dpi=170);plt.close(fig)

    result = dict(snapshot=str(snapshot), source=source['source'], single_training_seed=0,
                  validation_passed=True, runs=source['runs'], path_detail=details, path_history=histories,
                  dacer_on_comparison=comparisons, input_sha256=audit.INPUTS,
                  prior_summary_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest())
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# DACER OFF: 완료 결과와 진행 중 결과', '',
           'OptiQ, T=1, dense reward, NovelD OFF, 각 미로 seed0. 주 결과는 random-start direct-policy(random z + conditional sigma) 평가다. 성공/실패 모두 포함해 통로를 분류했다.', '',
           '|미로|상태|학습 step / 예산|평가 step|성공|실패 포함 통로|', '|---|---|---:|---:|---:|---|']
    for r in source['runs']:
        t=r['task']; m=details[t]['policy-natural']
        lines.append(f'|{t}|{r["status"]}|{r["step"]:,} / {r["budget"]:,}|{m["step"]:,}|{m["successes"]}/{m["episodes"]}|{m["counts"]}|')
    lines+=['', '## 완료된 정책: 평가 방식별 성공', '',
            '|미로|direct random|mu-only random|zero_z random|direct 동일 전체 상태|', '|---|---:|---:|---:|---:|']
    for r in source['runs']:
        if r['status']!='completed':continue
        def s(mode):
            m=r['modes'][mode];return f'{m["successes"]}/{m["episodes"]}'
        lines.append('|'+r['task']+'|'+'|'.join(s(k) for k in ('policy-natural','native-natural','zero_z-natural','policy-fixed'))+'|')
    lines+=['', '동일 상태 평가는 한 개의 선택된 전체 초기 상태를 반복한 것이다. 랜덤 시작점 전체의 성능과 같지 않다. 랜덤 시작점에서 두 통로를 쓰는 것과 같은 상태에서 두 통로로 나뉘는 것은 구분한다.',
            '', '## 완료된 DACER ON 대조군과 비교', '', '|미로|ON final 성공/100|OFF final 성공/100|ON return|OFF return|', '|---|---:|---:|---:|---:|']
    for r in source['runs']:
        if r['status']!='completed':continue
        on=prior['runs']['T=1'][r['task']]['modes']['policy-natural'];off=r['modes']['policy-natural']
        lines.append(f'|{r["task"]}|{on["successes"]}|{off["successes"]}|{on["mean_return"]:.1f}|{off["mean_return"]:.1f}|')
    lines+=['', '시드 하나 및 평가 100회 결과이므로 작은 차이를 일반적인 개선/악화로 단정하지 않는다. 설정 차이는 analysis.json에 전부 보존했다.',
            '', '## 이전에 완료된 dense + NovelD OFF 실험 요약', '', '아래는 각 미로 최종 direct-policy, random-start 100회 성공률(%). 방법마다 native sampling을 사용하며 원시 검증 결과를 재사용했다. 서로 다른 정책의 궤적을 합쳐 다양성을 계산하지 않았다.', '',
            '|방법/설정|v1|v2|v3|v4|', '|---|---:|---:|---:|---:|']
    for label,runs in prior['runs'].items():
        values=[str(runs[t]['modes']['policy-natural']['successes']) for t in ('v1','v2','v3','v4')]
        lines.append('|'+label+'|'+'|'.join(values)+'|')
    lines+=['', 'OptiQ의 위 이전 실험은 DACER ON이다. 현재 OFF 실험의 미완료 결과를 최종 비교표에 섞지 않았다.',
            '', '## 보존 및 검증', '',
            'config·평가 summary/원시 NPZ·checkpoint 검증 증명·result JSON을 로컬에 보존했다. 전송 SHA256, 원시 좌표/목표 도달/시작 상태/padding/유한성/dense 거리 보상 합을 검증했다. 완료 학습의 전체 replay/model checkpoint는 서버에 보존되며 서버 readback 및 dense replay 검증을 통과했다. 이 보고는 새 학습이나 rollout을 수행하지 않는다.', '',
            f'![전체 궤적]({out}/all_trajectories.png)', '', f'![학습곡선]({out}/learning_curves.png)', '', f'![통로 선택 추이]({out}/corridor_history.png)']
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    (out/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
    print(json.dumps(dict(output=str(out), summary={t:{m:{k:v for k,v in d.items() if k!='rows'} for m,d in modes.items()} for t,modes in details.items()}, config_differences={t:r['config_differences'] for t,r in comparisons.items()}),indent=2))


if __name__=='__main__':
    main()
