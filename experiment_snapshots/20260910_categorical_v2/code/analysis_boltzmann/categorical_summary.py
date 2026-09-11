"""Paired descriptive comparison against the preserved five-seed argmax runs."""
import argparse,base64,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .checkpoint_integrity import verify_initialization

def table(headers,rows):
    import html
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--require-complete',action='store_true');a=ap.parse_args();root=Path(a.campaign)
    protocol=json.loads((root/'protocol.json').read_text());base=Path(protocol['baseline']);tasks=json.loads((root/'tasks.json').read_text());out=root/'comparison';out.mkdir(exist_ok=True)
    history=[];rows=[];complete=[];available=[];init_checks=[]
    for task in tasks:
        cat=Path(task['out']);ref=base/'runs'/cat.name
        if not cat.exists():continue
        available.append(cat.name)
        if (cat/'actor_0.msgpack').exists():
            verify_initialization(task,mark=(cat/'COMPLETE').exists());init_checks.append(cat.name)
        done=(cat/'COMPLETE').exists() and (cat/'PAIRED_INITIALIZATION_OK').exists()
        if done:complete.append(cat.name)
        for variant,d in [('argmax',ref),('categorical',cat)]:
            for f in sorted(d.glob('distribution_*.npz')):
                with np.load(f) as x:
                    m=x['mode_mass'];target=x['reference_mode_mass'];samples=x['samples']
                    history.append(dict(case=task['case'],seed=task['seed'],initialization=task['initialization'],variant=variant,step=int(f.stem.split('_')[-1]),bin_tv=float(abs(m-target).sum()/2),effective_modes=float(1/(m@m)),coord_std=float(samples.std(0).mean())))
        if done:
            rr={variant:list(csv.DictReader((d/'estimates.csv').open())) for variant,d in [('argmax',ref),('categorical',cat)]}
            for k in [1,8,16,50,64,256,1024]:
                for method in ['optiq_raw','optiq_td']:
                    r={v:next(x for x in data if int(x['step'])==20000 and int(x['k'])==k and x['method']==method) for v,data in rr.items()}
                    assert abs(float(r['argmax']['truth'])-float(r['categorical']['truth']))<1e-12
                    row=dict(case=task['case'],seed=task['seed'],initialization=task['initialization'],k=k,method=method,truth=float(r['argmax']['truth']))
                    for v,x in r.items():
                        for key in ['mean','bias','rmse']:row[v+'_'+key]=float(x[key])
                    row['rmse_difference']=row['categorical_rmse']-row['argmax_rmse'];rows.append(row)
    summary=dict(completed=len(complete),expected=len(tasks),started=len(available),seeds=protocol['seeds'],initialization_hash_checks=len(init_checks),complete_runs=complete,paired_estimates=rows,mode_history=history)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,rr in [('paired_estimates',rows),('mode_history',history)]:
        if rr:
            with (out/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
    plots=[];cases=protocol['cases'];initlist=protocol['initializations'];colors=['#b54860','#267eac']
    if history:
        fig,axs=plt.subplots(2,3,figsize=(12,6.5))
        for ax,case in zip(axs.flat,cases):
            for ci,init in enumerate(initlist):
                for variant,ls in [('argmax','-'),('categorical','--')]:
                    rr=[r for r in history if r['case']==case and r['initialization']==init and r['variant']==variant]
                    steps=sorted({r['step'] for r in rr})
                    if not steps:continue
                    means=[np.mean([r['bin_tv'] for r in rr if r['step']==s]) for s in steps]
                    ax.plot(steps,means,ls=ls,marker='o',ms=3,color=colors[ci],label=f'{init} / {variant}')
            ax.set(xscale='symlog',xlim=(0,20000),ylim=(-.03,1.03),title=case,xlabel='Actor updates',ylabel='Mode-bin discrepancy');ax.grid(alpha=.15)
        axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'mode_history.png');fig.savefig(out/'mode_history.pdf');plt.close(fig);plots.append('mode_history')
    if rows:
        fig,axs=plt.subplots(1,2,figsize=(11,4.2));lab=['Backup RMSE','Mode-bin discrepancy']
        for ci,init in enumerate(initlist):
            rr=[r for r in rows if r['k']==50 and r['method']=='optiq_raw' and r['initialization']==init]
            if not rr:continue
            axs[0].scatter([r['argmax_rmse'] for r in rr],[r['categorical_rmse'] for r in rr],color=colors[ci],label=init,alpha=.7)
            pairs=[]
            for r in rr:
                pairs.append([next(x['bin_tv'] for x in history if x['case']==r['case'] and x['seed']==r['seed'] and x['initialization']==init and x['step']==20000 and x['variant']==v) for v in ['argmax','categorical']])
            axs[1].scatter(*np.array(pairs).T,color=colors[ci],label=init,alpha=.7)
        for ax,label in zip(axs,lab):
            lo,hi=min(ax.get_xlim()[0],ax.get_ylim()[0],0),max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],'k:',lw=1);ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Argmax',ylabel='Categorical',title=label+' (below diagonal is better)');ax.legend()
        fig.tight_layout();fig.savefig(out/'paired_final.png');fig.savefig(out/'paired_final.pdf');plt.close(fig);plots.append('paired_final')
    tr=[]
    for case in cases:
        for init in initlist:
            rr=[r for r in rows if r['case']==case and r['initialization']==init and r['method']=='optiq_raw' and r['k']==50]
            if not rr:continue
            seedset={r['seed'] for r in rr};hm={v:[r for r in history if r['case']==case and r['initialization']==init and r['variant']==v and r['step']==20000 and r['seed'] in seedset] for v in ['argmax','categorical']}
            tr.append([case,init,','.join(map(str,sorted(seedset))),f"{np.mean([r['argmax_rmse'] for r in rr]):.5f}",f"{np.mean([r['categorical_rmse'] for r in rr]):.5f}",f"{np.mean([r['rmse_difference'] for r in rr]):+.5f}",f"{sum(r['rmse_difference']<0 for r in rr)}/{len(rr)}",*[f"{np.mean([r['bin_tv'] for r in hm[v]]):.4f}" for v in ['argmax','categorical']]])
    im=''.join('<figure><img src="data:image/png;base64,'+base64.b64encode((out/(n+'.png')).read_bytes()).decode()+'"><figcaption>'+n+'</figcaption></figure>' for n in plots)
    header=f'<h1>OptiQ categorical target ablation</h1><p><strong>{len(complete)}/{len(tasks)} 완료 · seeds 0–4 · 20k updates · Sinkhorn 30회 · τ=0.25</strong></p>'
    text='''<p>기존 초기 actor와 optimizer checkpoint를 그대로 사용했다. 같은 coupling의 각 row에서 argmax 대신 categorical draw를 하며, 나머지 학습 설정은 유지했다. 기존 argmax 학습 결과는 보존했다.</p><p>아래 표는 완료된 categorical run과 정확히 짝지어진 argmax seed만 비교한다. ΔRMSE는 categorical−argmax로 음수이면 개선이다. 초기값의 hash 일치를 검사했다. 일부만 완료된 경우 문제 간 비교나 다섯 seed 전체 결론으로 해석하지 않는다.</p><p>Mode-bin 차이는 전체 밀도의 TV가 아니다. 30회 Sinkhorn의 row marginal 오차는 그대로 남으므로 categorical sampling이 weighted candidate marginal을 정확히 보존한다고 가정하지 않는다. 학습 결과의 원인과 return 개선은 별도 검증이 필요하다.</p>'''
    htable=table(['Case','Init','Seeds','Argmax RMSE','Categorical RMSE','Δ RMSE','Cat lower','Argmax bin TV','Cat bin TV'],tr) if tr else '<p>완료된 학습 결과 대기 중.</p>'
    style='body{font:15px/1.8 -apple-system,sans-serif;color:#253145;max-width:1120px;margin:40px auto;padding:25px;background:#fff}h1{font-size:29px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef2f7}img{width:100%}figure{margin:30px 0}figcaption{font-size:12px;color:#667}strong{color:#943850}'
    (out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>OptiQ categorical ablation</title><style>'+style+'</style><body>'+header+text+htable+im+'<p>Raw files: paired_estimates.csv · mode_history.csv · summary.json. This is a descriptive automatic report.</p></body></html>')
    print(json.dumps({k:summary[k] for k in ['completed','expected','started','initialization_hash_checks']}) ,flush=True)
    if a.require_complete:
        assert len(complete)==len(tasks),(len(complete),len(tasks))
        (root/'COMPLETE').write_text('All categorical runs and paired initialization checks complete.\n')

if __name__=='__main__':main()
