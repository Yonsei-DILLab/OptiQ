"""All-candidate audit plus diverse shortlist; no high-L submission logic."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .target import reference


def load(root,cases):
    all_results={}
    for case in cases:
        runs={}
        for seed in range(4):
            stage='screen' if seed<2 else 'validate_seeds'
            for method in ['forward','reverse']:
                p=root/'runtime'/stage/case['id']/f'{method}_s{seed}'
                if (p/'COMPLETE.json').exists():
                    c=json.loads((p/'COMPLETE.json').read_text());assert c['step']==100000
                    runs[f'{method}_s{seed}']={'metric':json.loads((p/'metrics_100000.json').read_text()),'folder':str(p)}
        all_results[case['id']]=runs
    return all_results


def gate(runs,seeds):
    if not all(f'{m}_s{s}' in runs for m in ['forward','reverse'] for s in seeds):return None
    return bool(all(runs[f'forward_s{s}']['metric']['forward_fitting_pass'] and runs[f'reverse_s{s}']['metric']['missing_modes']>=1 for s in seeds))


def shortlist(cases,data,seeds,max_count=6):
    good=[c for c in cases if gate(data[c['id']],seeds)]
    good.sort(key=lambda c:(c['family']!='reference',np.mean([data[c['id']][f'forward_s{s}']['metric']['histogram_TV'] for s in seeds])))
    selected=[];families=set()
    for c in good:
        if c['family'] not in families and len(selected)<max_count:selected.append(c);families.add(c['family'])
    for c in good:
        if c not in selected and len(selected)<max_count:selected.append(c)
    return selected


def plot_case(cfg,c,runs,output):
    x=np.linspace(-10,10,8193);target=reference(dict(cfg,**c),x)[0]
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),sharey=True)
    for j,(ax,method) in enumerate(zip(axes,['forward','reverse'])):
        values=[];tvs=[];missing=[];caption=''
        for seed in range(4):
            r=runs.get(f'{method}_s{seed}')
            if r:
                z=np.load(Path(r['folder'])/'samples_100000.npz');edges=z['edges'];values.append(z['histogram_mass']/np.diff(edges))
                tvs.append(r['metric']['histogram_TV']);missing.append(r['metric']['missing_modes'])
        if values:
            y=np.mean(values,axis=0);ax.stairs(y,edges,fill=True,alpha=.18,color=f'C{j}',linewidth=0)
            ax.stairs(y,edges,color=f'C{j}',lw=1.2,label=f'{method.capitalize()} ({len(values)} seeds)')
            caption=f'Mean seed TV = {np.mean(tvs):.3f}; missing modes: {missing}'
        ax.plot(x,target,'k--',lw=1.3,label='Target');ax.set(xlim=(-10,10),xlabel='Action\n'+caption,title=f'{method.capitalize()} KL');ax.legend(loc='upper right',fontsize=8)
    axes[0].set_ylabel('Probability density');fig.suptitle(c['id']+' | '+str(len(c['target_centers']))+' modes',fontsize=13)
    fig.tight_layout();fig.savefig(output/f'{c["id"]}.png',dpi=180);plt.close(fig)
    seeds=[s for s in range(4) if any(f'{m}_s{s}' in runs for m in ['forward','reverse'])]
    if not seeds:return
    fig,axes=plt.subplots(len(seeds),2,figsize=(11,2.5*len(seeds)),squeeze=False)
    for ri,seed in enumerate(seeds):
        for mi,method in enumerate(['forward','reverse']):
            ax=axes[ri,mi];r=runs.get(f'{method}_s{seed}')
            if r:
                z=np.load(Path(r['folder'])/'samples_100000.npz');y=z['histogram_mass']/np.diff(z['edges'])
                ax.stairs(y,z['edges'],fill=True,alpha=.18,color=f'C{mi}');ax.stairs(y,z['edges'],color=f'C{mi}',lw=1)
                ax.set_title(f'{method} seed {seed} | TV {r["metric"]["histogram_TV"]:.3f} | missing {r["metric"]["missing_modes"]}')
            else:ax.set_title(f'{method} seed {seed}: pending')
            ax.plot(x,target,'k--',lw=1);ax.set_xlim(-10,10)
    fig.tight_layout();fig.savefig(output/f'{c["id"]}_all_seeds.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',default='config.json',choices=['config.json','shape_config.json']);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(__file__).with_name(args.config).read_text());cases=cfg['cases'];data=load(args.root,cases)
    discovery=[c['id'] for c in cases if gate(data[c['id']],[0,1])];selected=shortlist(cases,data,[0,1,2,3])
    plt.style.use('default');plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=(len(cases)+3)//4;fig,axes=plt.subplots(rows,4,figsize=(14,2.7*rows),squeeze=False);x=np.linspace(-10,10,4097)
    for ax,c in zip(axes.flat,cases):
        ax.plot(x,reference(dict(cfg,**c),x)[0],'k-',lw=1.2);ax.set_title(c['id'],fontsize=9);ax.set_xlim(-10,10);ax.tick_params(labelsize=8)
    
    for ax in list(axes.flat)[len(cases):]:ax.set_visible(False)
    fig.suptitle('Target-only candidate grid (not training results)');fig.tight_layout();fig.savefig(args.output/'target_candidates.png',dpi=180);plt.close(fig)
    for c in cases:
        if data[c['id']]:plot_case(cfg,c,data[c['id']],args.output)
    payload={'high_L_status':'NOT_AUTHORIZED_NOT_SUBMITTED','completed_runs':sum(map(len,data.values())),
             'screen_pass_ids':discovery,'four_seed_selected_ids':[c['id'] for c in selected],
             'all_results':data,'config':cfg}
    (args.output/'SUMMARY.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Diverse target: Forward / Reverse KL 후보 탐색','',
      f'완료 {payload["completed_runs"]}개 run. 모든 실행은 100K updates, Reverse L=1024이다. **L=2²⁰ 본 실험은 사용자 확인 전이며 제출하지 않았다.**','',
      '## 공통 설정','',
      'N=M128, batch32, Adam3e-4, temperature0.25, mean-head 초기화 scale3, log sigma[-5,-1], action[-10,10]. Actor·optimizer·proposal은 모든 target에서 같다. 변경한 것은 mode 수, 중심, 폭, 질량뿐이다. Paired methods는 같은 GPU에서 동일 초기값으로 시작했다.','',
      r'$$f(a)=\sum_k\rho_k\mathcal N(a;c_k,h_k^2),\qquad Q(a)=0.25\log f(a),\qquad p^\star(a)=f(a)/\int_{-10}^{10}f(x)dx.$$','',
      '학습은 Q/gradient만 사용하며 정답 target sample과 component label은 제공하지 않는다. 최종 histogram은 각 seed에서 실제 2¹⁸ action,512 bins로 측정했다. KDE는 없다. 평균 그림 외에 모든 개별 seed 그림도 남긴다. Basin은 정답 density의 골짜기로 나누고 core는 peak ± component std이다. Mode missing은 core와 basin 질량이 모두 정답의25% 미만인 경우다. 후보는 Forward의 모든 peak 복구 및 TV≤.15, Reverse의 최소1mode 누락이 네 seed 모두에서 유지되어야 한다.','',
      '이 자료는 차이가 드러나는 예시를 찾는 탐색 결과다. 전체 표의 실패·반대 결과도 보존했다. Mode 복구는 density가 완벽히 동일하다는 뜻이 아니며 실제 TV를 함께 보고한다.','',
      '폭·질량이 다른 그림형 후보의 질량 표기는 box 조건부 질량이다. 전체 Gaussian 혼합계수와 구분하며 SHAPE_PROTOCOL.md에 변환식을 기록했다.' if args.config=='shape_config.json' else '', '', '## 전체 target 후보','', '![목표분포 후보](target_candidates.png)','',
      '|ID|Mode 수|중심|폭|질량|','|---|---:|---|---|---|']
    for c in cases:lines.append(f'|{c["id"]}|{len(c["target_centers"])}|{c["target_centers"]}|{c["target_widths"]}|{[round(x,3) for x in c.get("box_component_masses",c["target_masses"])]}|')
    lines+=['','## 네 seed를 통과한 후보 (최대6개)','']
    if not selected:lines+=['아직 네 seed 검증을 마친 통과 후보가 없다. 아래의 완료 결과는 탐색 중간 결과이다.','']
    for c in selected:lines+=[f'### {c["id"]}', '',f'![평균]({c["id"]}.png)','',f'[개별 seed]({c["id"]}_all_seeds.png)','']
    lines+=['## 모든 완료 결과','', '|ID|0·1 pass|0–3 pass|Forward TV (seed순)|Reverse TV (seed순)|Reverse 누락 수|','|---|---|---|---|---|---|']
    for c in cases:
        r=data[c['id']];vals=lambda method,key:[round(r[f'{method}_s{s}']['metric'][key],4) if f'{method}_s{s}' in r else None for s in range(4)]
        lines.append(f'|{c["id"]}|{gate(r,[0,1])}|{gate(r,list(range(4)))}|{vals("forward","histogram_TV")}|{vals("reverse","histogram_TV")}|{vals("reverse","missing_modes")}|')
    for c in cases:
        if data[c['id']]:lines+=['',f'### {c["id"]}', '',f'![平均]({c["id"]}.png)','',f'[모든 seed]({c["id"]}_all_seeds.png)']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:payload[k] for k in ['completed_runs','screen_pass_ids','four_seed_selected_ids','high_L_status']},indent=2))

if __name__=='__main__':main()
