"""Six-target appendix from final paired runs and frozen-score measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..kl_paper_highL.render import style, save, limits, score_axis, score_legend, BLUE, ORANGE
from ..kl_diverse_targets_1d.target import reference as gm_reference
from ..kl_nongmm_targets_1d.target import reference as ng_reference

CASES = [
    ('t00_reference', 'Three Gaussian modes'),
    ('n00_spike_ramp', 'Spike + ramp'),
    ('n07_spike_flat_ramp', 'Spike + plateau + ramp'),
    ('t01_two_offset', 'Two offset Gaussian modes'),
    ('t05_unequal_mass', 'Unequal Gaussian masses'),
    ('t06_minor_mode', 'Minor Gaussian mode'),
]
SEEDS = {case: list(range(3 if case == 't06_minor_mode' else 4)) for case, _ in CASES}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load(root):
    density, scores, rows, score_rows, provenance = {}, {}, [], [], {
        'density_runs': [], 'score_runs': [], 'incomplete_runs': [], 'paired_seeds': SEEDS,
        'score_main_seed': 0, 'steps': 100000, 'reverse_L': 2**20, 'bins': 512, 'samples_per_seed': 2**20}
    for case, name in CASES:
        for method in ['forward', 'reverse']:
            used = []
            for seed in range(4):
                folder = root/'parent/runtime/confirm'/case/f'{method}_s{seed}'
                if not (folder/'COMPLETE.json').exists():
                    assert (case, method, seed) == ('t06_minor_mode', 'reverse', 3)
                    provenance['incomplete_runs'].append(dict(case=case, method=method, seed=seed,
                        status=read(folder/'STATUS.json'), checkpoint_sha256=sha(folder/'checkpoint.msgpack')))
                    continue
                run = read(folder/'RUN.json'); metric = read(folder/'metrics_100000.json')
                assert read(folder/'COMPLETE.json')['step'] == metric['step'] == 100000
                assert metric['sample_count'] == 2**20
                assert run['config']['n'] == run['config']['m'] == 128 and run['config']['batch'] == 32
                assert run['L'] == (2**20 if method == 'reverse' else 0)
                paired = seed in SEEDS[case]
                if paired:
                    other = read(root/'parent/runtime/confirm'/case/f'{"forward" if method=="reverse" else "reverse"}_s{seed}'/'RUN.json')
                    assert run['initial_parameter_sha256'] == other['initial_parameter_sha256']
                with np.load(folder/'samples_100000.npz') as d:
                    edges = d['edges'].copy(); mass = d['histogram_mass'].copy()
                    assert len(d['actions']) == 2**20 and len(mass) == 512
                    assert np.allclose(np.histogram(d['actions'], edges)[0]/2**20, mass)
                    assert abs(.5*np.abs(mass-d['target_mass']).sum()-metric['histogram_TV']) < 1e-8
                if paired:
                    used.append(mass/np.diff(edges))
                rows.append(dict(case=case, method=method, seed=seed, included_in_paired_figure=paired,
                    TV=metric['histogram_TV'], W1=metric['wasserstein_1'], missing_modes=metric['missing_modes'],
                    mode_mass=metric['mode_mass']))
                provenance['density_runs'].append(dict(case=case, method=method, seed=seed,
                    included_in_paired_figure=paired, run=run, metrics=metric,
                    checkpoint_sha256=sha(folder/'checkpoint.msgpack'), samples_sha256=sha(folder/'samples_100000.npz')))
            grid = np.linspace(-10, 10, 16385)
            ref = ng_reference if run['config']['target_kind']=='nongmm' else gm_reference
            density[case, method] = dict(edges=edges, values=np.asarray(used), x=grid,
                target=ref(run['config'],grid)[0], config=run['config'])
        for seed in SEEDS[case]:
            folder=root/'runtime/results'/case/f'reverse_s{seed}'
            done=read(folder/'COMPLETE.json'); run=read(folder/'RUN.json'); summary=read(folder/'SUMMARY.json')
            assert done['checkpoint_sha256'] == sha(root/'parent/runtime/confirm'/case/f'reverse_s{seed}'/'checkpoint.msgpack')
            with np.load(folder/'scores.npz') as d: data={k:d[k].copy() for k in d.files}
            assert data['scores'].shape==(16,18,140) and data['reference_scores'].shape==(4,140)
            assert np.isfinite(data['scores']).all() and np.array_equal(data['Ls'],2**np.arange(7,25))
            scores[case,seed]=data
            for rec in summary['results']:score_rows.append(dict(case=case,seed=seed,**rec))
            provenance['score_runs'].append(dict(case=case,seed=seed,run=run,complete=done,summary=summary,
                raw_sha256=sha(folder/'scores.npz'),validation=read(folder/'VALIDATION.json')))
    assert len(rows)==47 and len(provenance['score_runs'])==23
    return density,scores,rows,score_rows,provenance


def density_axis(ax, data, method, title, ylabel=False):
    color=BLUE if method=='forward' else ORANGE
    for y in data['values']:ax.stairs(y,data['edges'],color=color,alpha=.18,lw=.65)
    mean=data['values'].mean(0)
    ax.stairs(mean,data['edges'],fill=True,color=color,alpha=.20,lw=0)
    ax.stairs(mean,data['edges'],color=color,lw=1.55)
    ax.plot(data['x'],data['target'],'k--',lw=1.45)
    ax.set(xlim=(-10,10),ylim=(0,None),xlabel=r'$a$')
    ax.set_xticks([-10,-5,0,5,10]);ax.tick_params(direction='out',length=4,width=.8)
    if ylabel:ax.set_ylabel('Density')
    ax.set_title(title,pad=12)


def figures(density,scores,out):
    fig,axes=plt.subplots(3,4,figsize=(18.8,11.8))
    fig.subplots_adjust(left=.048,right=.995,bottom=.065,top=.945,wspace=.30,hspace=.49)
    for n,(case,_) in enumerate(CASES):
        for k,method in enumerate(['forward','reverse']):
            j=2*n+k
            density_axis(axes[j//4,j%4],density[case,method],method,
                         f'({chr(97+j)}) {method.capitalize()} KL',j%4==0)
    save(fig,out,'density_six_3x4')

    # Six rows for a single overview, plus two three-row pages for paper layout.
    for zoom in [True,False]:
        suffix='zoom' if zoom else 'common_y'
        for group,label in [(list(range(6)),'all'),(list(range(3)),'page1'),(list(range(3,6)),'page2')]:
            fig,axes=plt.subplots(len(group),3,figsize=(14.3,3.85*len(group)))
            fig.subplots_adjust(left=.075,right=.99,bottom=.095 if len(group)==3 else .05,
                                top=.955 if len(group)==3 else .975,hspace=.70,wspace=.33)
            for row,n in enumerate(group):
                case,name=CASES[n];data=scores[case,0]
                for j in range(3):
                    score_axis(axes[row,j],data,j,f'({chr(97+3*n+j)}) {[10,50,90][j]}% policy quantile',
                        limits(data,[128+j] if zoom else [128,129,130]),j==0)
            score_legend(fig)
            save(fig,out,f'score_six_{label}_{suffix}')

    # Per-case density and score in one row; also an all-case combined overview.
    for group,label in [([n],case) for n,(case,_) in enumerate(CASES)]+[(list(range(6)),'six')]:
        fig,axes=plt.subplots(len(group),5,figsize=(24,4.05*len(group)),squeeze=False)
        fig.subplots_adjust(left=.04,right=.995,bottom=.20 if len(group)==1 else .035,
                            top=.80 if len(group)==1 else .975,hspace=.72,wspace=.40)
        for row,n in enumerate(group):
            case,name=CASES[n]
            for j,method in enumerate(['forward','reverse']):
                density_axis(axes[row,j],density[case,method],method,f'({chr(97+j)}) {method.capitalize()} KL',j==0)
            for j in range(3):
                data=scores[case,0]
                score_axis(axes[row,j+2],data,j,f'({chr(99+j)}) {[10,50,90][j]}% policy quantile',limits(data,[128+j]),j==0)
        save(fig,out,f'density_and_score_{label}')


def report(out,root,rows,score_rows,provenance):
    tables=[]; errors=[]
    for case,name in CASES:
        for method in ['forward','reverse']:
            r=[x for x in rows if x['case']==case and x['method']==method and x['included_in_paired_figure']]
            tv=np.array([x['TV'] for x in r]);w1=np.array([x['W1'] for x in r])
            tables.append(f'| {name} | {method.capitalize()} | {len(r)} | {tv.mean():.4f} ± {tv.std(ddof=1):.4f} | {w1.mean():.4f} ± {w1.std(ddof=1):.4f} | {[x["missing_modes"] for x in r]} |')
        r=[x for x in score_rows if x['case']==case and x['L']==2**20]
        e=[x['policy_relative_RMSE_mean']*100 for x in r]
        errors.append(f'| {name} | {len(r)} | {min(e):.3f}–{max(e):.3f}% |')
    text=r'''# Six-target appendix: density and score convergence

기존에 선정한 환경은 **여섯 개**다. 이 자료는 Reverse $L=2^{20}$, 100K updates의 결과만 사용한다. Main의 두 환경 그림은 그대로 보존했다.

**완료 범위:** 48개 예정 평가 중 47개가 완료됐다. `t06_minor_mode/reverse_s3`만 백업 상태가 82,935 updates이며 원래 Vast SSH는 현재 연결을 거부한다. 이 환경은 완료된 **동일 seed 0–2를 Forward와 Reverse 양쪽에 사용**했다. 나머지 다섯 환경은 seed 0–3이다. 제외 이유는 미완료이며, 결과에 따른 seed 선택이 아니다. Forward seed 3의 완료 결과도 `density_metrics.csv`에 보존했다. 이번 작업에서는 추가 학습을 실행하지 않았다.

## 1. 환경 목록

모든 action은 $[-10,10]$, $Q=0.25\log p^\star$, $N=M=128$, batch 32, Adam $3\times10^{-4}$, mean-head initializer scale 3, log-sigma 범위 $[-5,-1]$이다. 상세 수식과 actor/gradient 구현은 [논문용 appendix](appendix.md)에 있다.

| 순서 | ID | Target |
|---|---|---|
| 1 | t00_reference | Gaussian 중심 $(-4.25,0,4.25)$, std 0.5, 질량 $(1/3,1/3,1/3)$ |
| 2 | n00_spike_ramp | Logistic spike + rising ramp, 질량 $(0.2,0.8)$ |
| 3 | n07_spike_flat_ramp | Logistic spike + plateau + rising ramp, 질량 $(0.25,0.30,0.45)$ |
| 4 | t01_two_offset | Gaussian 중심 $(0,4.25)$, std 0.5, 질량 $(0.5,0.5)$ |
| 5 | t05_unequal_mass | Gaussian 중심 $(-4.25,0,4.25)$, std 0.5, 질량 $(0.2,0.5,0.3)$ |
| 6 | t06_minor_mode | Gaussian 중심 $(-4.25,0,4.25)$, std 0.5, 질량 $(0.1,0.65,0.25)$ |

## 2. 전체 density: 3행 × 4열

![Six densities](density_six_3x4.png)

왼쪽에서 오른쪽, 위에서 아래로 Forward–Reverse 두 패널씩 위 표의 환경 1–6이다. 검정 점선은 정답 target, 파랑은 Forward, 주황은 Reverse다. 실제 action **$2^{20}$개/seed**, **512-bin histogram**, smoothing 없이 seed 평균을 그렸다. 옅은 가는 선은 개별 seed, 채움은 density 아래 면적이다. 각 패널 y축은 독립적이다. 마지막 두 패널만 3-seed 평균이다.

Histogram TV는 $\frac12\sum_b|\hat p_b-p_b^\star|$이다. 예를 들어 0.1이면 bin별 확률질량을 맞추기 위해 총 질량의 10%를 이동해야 한다. Continuous density의 정확한 TV가 아니라 binning한 분포의 TV다. 1D Wasserstein-1은 $W_1=\int_{-10}^{10}|\hat F(a)-F^\star(a)|\,da$로 계산하며 action 좌표와 같은 단위다.

Missing mode는 target의 골짜기로 나눈 basin과 사전에 지정한 core 모두에서 actor mass가 해당 target mass의 25% 미만인 경우다. Gaussian core는 각 peak의 ±0.5이다. Spike+ramp core는 $[-4.5,-4]$, $[3,7.7]$이고, spike+plateau+ramp core는 $[-5.25,-4.75]$, $[-0.75,0.75]$, $[4,6.8]$이다.

| Target | Method | Paired seeds | TV mean ± seed SD | W1 mean ± seed SD | Missing modes by seed |
|---|---|---:|---:|---:|---|
{{TABLE}}

TV/W1은 seed별 오차의 평균이다. Target들이 screening에서 선정된 사례임을 명시해야 하며, 작은 Forward fitting 오차도 그대로 남겨 두었다. 완벽한 복구나 모든 target에서의 우열을 주장하지 않는다.

## 3. 전체 score 수렴

![Score first three](score_six_page1_zoom.png)

위 그림의 행은 환경 1–3, 아래 그림은 환경 4–6이다. 각 행은 seed 0 Reverse policy의 10%, 50%, 90% 분위수 action을 고정했다. 파란 실선은 16회 MC 평균, 음영은 반복값의 10–90% 구간이다. 가로 점선은 별도 $2^{24}$ MC samples로 4회 추정한 평균, 세로 파선은 학습의 $L=2^{20}$이다. Gray reference band는 없다. **Number of MC samples $L$은 latent MC sample 수**이며 policy action 수 $M$, 평가용 action sample 수, 반복 횟수와 다르다. 확대 버전은 패널별 y축 범위를 사용한다.

![Score last three](score_six_page2_zoom.png)

최종 Reverse actor 23개를 모두 평가했다. 기존 두 환경의 8개 score 측정은 재사용했고, 나머지 15개만 새로 측정했다. 모든 checkpoint hash를 대조했다. $L=2^7,\ldots,2^{24}$; 16회 독립 반복 안에서 L별 prefix를 공유하고, 별도의 기준 추정을 4회 수행한다. Actor는 float32, 안정화한 비율 누산은 float64를 사용한다.

128개 고정 policy action에서 큰-MC 기준 대비 relative RMSE를 계산했다. 각 반복의 score RMSE를 기준 score의 RMS로 나누고 16회 평균했다. 아래는 seed 사이의 범위다.

| Target | Evaluated Reverse seeds | Relative RMSE at $L=2^{20}$ |
|---|---:|---:|
{{ERRORS}}

큰-MC 기준 자체도 근사다. 위 결과는 최종 actor가 방문하는 영역에서의 경험적 안정성이고, 모든 학습 시점이나 사라진 mode 주변에서의 정확성 보장은 아니다. 추가 target probe의 오차도 `score_metrics.csv`에 포함했다.

## 4. Density와 score를 함께 보는 버전

[전체 6행×5열 PDF](density_and_score_six.pdf)에서 각 행은 같은 환경이며, Forward density → Reverse density → score 10% → 50% → 90% 순서다. Score의 단위는 density와 다르므로 서로 다른 y축으로 읽는다. 패널문자(a–e)는 행 안의 종류를 가리킨다. 논문 배치에 맞춰 각 환경을 1행짜리 파일로도 저장했다.

{{PER_CASE}}

## 5. 파일과 보관

- Density: [PDF](density_six_3x4.pdf) · [SVG](density_six_3x4.svg)
- Score 확대: [전체 6행 PDF](score_six_all_zoom.pdf) · [환경 1–3 PDF](score_six_page1_zoom.pdf) · [환경 4–6 PDF](score_six_page2_zoom.pdf)
- Score 공통 y축: [전체 6행 PDF](score_six_all_common_y.pdf) · [환경 1–3 PDF](score_six_page1_common_y.pdf) · [환경 4–6 PDF](score_six_page2_common_y.pdf)
- [논문 appendix](appendix.md) · [영문 captions](captions.md)
- [모든 완료 seed의 density 수치](density_metrics.csv) · [모든 L의 score 수치](score_metrics.csv) · [Provenance](PROVENANCE.json)

공통 y축 버전은 같은 환경의 세 action에 동일 범위를 적용한다. 새로운 score 측정 commit: `{{MEASUREMENT}}`. 그림·문서 commit: `{{RENDER}}`. 기존 두 환경의 score commit은 `fa630a1f43f9303d4f441b9f2feeb9c1fb935c4e`이다. Raw run별 training/measurement SHA는 provenance에 기록했다.

로컬: `reports/20260926_kl_appendix_six/`, `studies/20260926_kl_appendix_six/`.
중앙 보관: `dildata:/data1/heejoonorm/OptiQ/reports/20260926_kl_appendix_six/`, `dildata:/data1/heejoonorm/OptiQ/studies/20260926_kl_appendix_six/`.
'''
    links='\n'.join(f'- {i+1}. {name}: [PNG](density_and_score_{case}.png) · [PDF](density_and_score_{case}.pdf)' for i,(case,name) in enumerate(CASES))
    for a,b in {'TABLE':'\n'.join(tables),'ERRORS':'\n'.join(errors),'PER_CASE':links,
                'MEASUREMENT':provenance['measurement_commit'],'RENDER':provenance['render_commit']}.items():text=text.replace('{{'+a+'}}',b)
    assert '{{' not in text
    (out/'report.md').write_text(text)
    (out/'appendix.md').write_text(Path(__file__).with_name('appendix.md').read_text())
    (out/'captions.md').write_text(r'''# Figure captions

**Forward- and reverse-KL fitting across six selected targets.** Forward and Reverse density panels are paired for, in order: three equally weighted Gaussian modes; spike and ramp; spike, plateau, and ramp; two offset Gaussian modes; unequal Gaussian masses; and a minor Gaussian mode. Black dashed curves show the target density. Colored curves average 512-bin histograms from $2^{20}$ policy samples per seed after 100,000 updates, without smoothing. Light fills show density, and faint traces show individual seeds. The first five targets use four paired seeds; the final target uses paired seeds 0–2 because Reverse seed 3 did not complete. Reverse uses $L=2^{20}$ latent MC samples during training. Vertical scales differ between panels. Targets were selected during prior screening.

**Monte Carlo convergence of the reverse-policy action score.** Each row corresponds to the same target order, and each column fixes the 10th, 50th, or 90th percentile action of its final seed-0 Reverse policy. The horizontal axis is the number of latent MC samples $L$. Blue curves show the mean of 16 independent MC repetitions, with 10th–90th percentile shading. Prefixes are nested across $L$ within each repetition. Dotted horizontal lines are empirical references averaged over four additional independent estimates with $2^{24}$ samples each; vertical dashed lines mark training $L=2^{20}$. The reference is not exact. Zoomed panels use individual vertical ranges; common-y panels share a range within each target.

**Joint density and score view.** Each row contains Forward density, Reverse density, and the three fixed-action score-convergence plots for one target. Density averages paired training seeds while scores display seed 0. The same data and conventions as above apply. Panel letters (a–e) identify the five plot types within each row.
''')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--render-commit',required=True);a=p.parse_args()
    root=a.workspace/'studies/20260926_kl_appendix_six';out=a.workspace/'reports/20260926_kl_appendix_six';out.mkdir(parents=True,exist_ok=True)
    style();density,scores,rows,score_rows,provenance=load(root)
    provenance.update(render_commit=a.render_commit,measurement_commit=read(root/'SOURCE_MANIFEST.json')['commit'])
    figures(density,scores,out);csv_write(out/'density_metrics.csv',rows);csv_write(out/'score_metrics.csv',score_rows)
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    report(out,root,rows,score_rows,provenance);print(out)


if __name__=='__main__':main()
