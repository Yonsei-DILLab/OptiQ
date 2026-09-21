"""Readable provisional/final report: actual sample densities and adaptation curves."""
from pathlib import Path
import argparse,json,html,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nsq.config import METHODS,EPSILONS,DIMS,SIZES

COLORS={'gmm_learned':'#e65d36','exact_learned':'#7957b3','argmax_truncated':'#274d71'}
def metrics(run,root,task):
 folders=([root/'runs'/task['parent']] if task.get('parent') else [])+[run]
 data={}
 for folder in folders:
  for p in sorted((folder/'evaluations').glob('*.npz')):
   z=np.load(p);steps=z['step']
   for i,step in enumerate(steps):data[int(step)]={k:float(z[k][i]) for k in z.files}
 return data

def main(root):
 root=Path(root);out=root/'report';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
 tasks=json.loads((root/'tasks.json').read_text());comparisons=[t for t in tasks if t['stage'] not in ['prefix','source']]
 complete=[t for t in comparisons if (root/'runs'/t['name']/'COMPLETE.json').exists()]
 lines=['# Non-stationary Q: 어떤 방법이 가장 빨리 적합하는가?',
  '**기존 결과를 덮어쓰지 않는 새 실험이다. 미완료 결과는 최종 우열로 해석하지 않는다.**',
  f'완료 비교 궤적 {len(complete)}/{len(comparisons)}. 네 차원·네 크기·4 seeds. 상세 설정은 [PROTOCOL](../PROTOCOL.md).',
  '## 그림 읽는 법',
  '모든 actor density는 32,768개 실제 action histogram이다. 2D는 64×64 joint histogram을 3D surface와 heatmap으로 표시한다. 4D·8D에서는 첫 좌표와 다른 좌표를 분리한다. Marginal TV는 joint TV가 아니다.',
  '작은 Sinkhorn ε에서 marginal이 맞지 않으면 수렴한 OT와 구분한다. 표에는 absolute TV, 변화 전 대비 오차,학습 compute 시간과 진단 포함 wall time을 함께 제공한다. 서로 다른 GPU의 시간은 직접 알고리즘 속도 차이로 해석하지 않는다.',
  'D>1 learned-Q reference는 Sobol importance quadrature이다. ESS<512 또는 두 반쪽 reference의 marginal TV>0.05이면 해당 reference 지표를 신뢰하기 어렵다고 표시한다.',
  '## 진행 상황','|단계|완료|전체|','|---|---:|---:|']
 for stage in ['prefix','mass','split','source','replay','closed']:
  group=[t for t in tasks if t['stage']==stage];done=sum((root/'runs'/t['name']/'COMPLETE.json').exists() for t in group)
  lines.append(f'|{stage}|{done}|{len(group)}|')
 summaries=[]
 for family,stage in [('double','mass'),('tri','mass'),('tri','split'),('tri','replay'),('tri','closed')]:
  for dim in DIMS:
   for n,m in SIZES:
    group=[t for t in comparisons if (t['family'],t['stage'],t['dim'],t['n'],t['m'])==(family,stage,dim,n,m)]
    available=[t for t in group if list((root/'runs'/t['name']/'evaluations').glob('*.npz'))]
    if not available:continue
    key=f'{family}_{stage}_D{dim}_N{n}_M{m}';lines+=['',f'## {family}/{stage} · {dim}D · {n}×{m}']
    readings={t['name']:metrics(root/'runs'/t['name'],root,t) for t in available}
    fig,axes=plt.subplots(3,4,figsize=(17,10),sharex=True,sharey=True)
    for ax,eps in zip(axes.flat,EPSILONS):
     for method in ['gmm_learned','exact_learned','argmax_truncated','sinkhorn_e'+format(eps,'g')]:
      rs=[readings[t['name']] for t in available if t['method']==method]
      common=sorted(set.intersection(*[set(x) for x in rs])) if rs else []
      if not common:continue
      ys=np.array([[r[s]['first_axis_tv'] for s in common] for r in rs]);mean=ys.mean(0)
      ax.plot(common,mean,label=method,color=COLORS.get(method,'#078d9e'))
      if len(rs)>1:ax.fill_between(common,mean-ys.std(0,ddof=1),mean+ys.std(0,ddof=1),alpha=.10,color=COLORS.get(method,'#078d9e'))
     ax.set_title(f'Sinkhorn epsilon={eps:g}');ax.set_ylim(0,1);ax.grid(alpha=.2)
     if stage in ['mass','split']:
      for x in [20000,25000,30000]:ax.axvline(x,color='gray',ls=':',lw=.7)
    axes[0,0].legend(fontsize=7);fig.supxlabel('Actor updates');fig.supylabel('First-coordinate histogram TV');fig.suptitle(key+' | seed mean ± SD; partial runs shown');fig.tight_layout();fig.savefig(figs/(key+'_tracking.png'),dpi=150);plt.close(fig)
    lines.append(f'![epsilon sweep]({"figures/"+key+"_tracking.png"})')
    for t in available:
     r=readings[t['name']];steps=sorted(r)
     last=r[steps[-1]];row=dict(family=family,stage=stage,dim=dim,n=n,m=m,method=t['method'],seed=t['seed'],last_step=steps[-1],complete=(root/'runs'/t['name']/'COMPLETE.json').exists(),**last)
     for boundary in ([20000,25000,30000] if stage=='mass' else [20000,27000] if stage=='split' else [0]):
      pre=[r[s]['first_axis_tv'] for s in steps if boundary-1000<=s<=boundary]
      baseline=float(np.mean(pre)) if pre else np.nan
      for horizon in [100,500,1000,5000]:
       ts=[s for s in steps if boundary<=s<=boundary+horizon]
       # Integrals are only reported when the entire requested window is observed.
       if len(ts)>1 and ts[0]==boundary and ts[-1]==boundary+horizon:
        row[f'auc_{boundary}_{horizon}']=float(np.trapz([r[s]['first_axis_tv'] for s in ts],ts)/horizon)
      for label,threshold in [('absolute01',.1),('baseline_plus005',baseline+.05)]:
       post=[s for s in steps if s>boundary and s<=boundary+5000];hit=None
       for i in range(max(0,len(post)-2)):
        if all(r[s]['first_axis_tv']<=threshold for s in post[i:i+3]):hit=post[i];break
       row[f'recovery_{boundary}_{label}']=None if hit is None else hit-boundary
     summaries.append(row)
    # Every method's final/most recent direct histogram, seed0 (preselected).
    examples=[]
    for t in available:
     if t['seed']!=0:continue
     rd=root/'runs'/t['name']/'density';p=rd/'final_independent.npz'
     if not p.exists():
      files=sorted(rd.glob('[0-9]*.npz'))
      if not files:continue
      p=files[-1]
     examples.append((t,np.load(p)))
    fig,axes=plt.subplots(int(np.ceil(len(examples)/4)),4,figsize=(18,max(3,3*np.ceil(len(examples)/4))),squeeze=False)
    for ax,(t,z) in zip(axes.flat,examples):
     centers=(z['edges'][1:]+z['edges'][:-1])/2;dx=np.diff(z['edges'])
     ax.plot(centers,z['target_marginals'][0]/dx,'--',c='#283448',label='target first coordinate')
     ax.plot(centers,z['marginal_mass'][0]/dx,c='#078d9e',label='actor histogram first')
     if dim>1:ax.plot(centers,z['marginal_mass'][1:].mean(0)/dx,c='#e65d36',alpha=.7,label='actor transverse mean')
     ax.set_title(t['method']+f' | update {int(z["step"])}',fontsize=9);ax.set_xlim(-1,1);ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(examples):]:ax.axis('off')
    axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figs/(key+'_densities.png'),dpi=150);plt.close(fig)
    lines.append(f'![actual sample histograms](figures/{key}_densities.png)')
    if dim==2:
     for t,z in examples:
      if t['method'] not in ['gmm_learned','exact_learned','argmax_truncated','sinkhorn_e0.1']:continue
      edges=z['joint_edges'];mid=(edges[1:]+edges[:-1])/2;X,Y=np.meshgrid(mid,mid,indexing='ij');area=np.diff(edges)[0]**2
      fig=plt.figure(figsize=(13,10))
      for col,(label,mass) in enumerate([('Target joint density',z['target_joint_mass']),('32768-action histogram',z['joint_mass'])]):
       ax=fig.add_subplot(2,2,col+1,projection='3d');ax.plot_surface(X,Y,mass/area,cmap='viridis',linewidth=0);ax.set(xlabel='a1',ylabel='a2',zlabel='density',title=label)
       ax=fig.add_subplot(2,2,col+3);ax.imshow((mass/area).T,origin='lower',extent=(-1,1,-1,1),cmap='viridis');ax.set(xlabel='a1',ylabel='a2',title=label)
      fig.suptitle(key+' | '+t['method']);fig.tight_layout();path=key+'_'+t['method']+'_3d.png';fig.savefig(figs/path,dpi=150);plt.close(fig);lines.append(f'![2D joint distribution](figures/{path})')
    for _,z in examples:z.close()
 fields=sorted(set().union(*(r.keys() for r in summaries))) if summaries else ['method','seed','complete']
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summaries)
 lines+=['','## 完整原始记录 / 원본 기록','[전체 수치](summary.csv). 성공/실패·미회복 기록을 함께 유지한다.']
 md='\n\n'.join(lines)+'\n';(out/'report.md').write_text(md)
 # Simple dependency-free HTML; math/protocol remains linked as Markdown.
 import re
 chunks=[]
 for part in lines:
  if part.startswith('!['):
   match=re.match(r'!\[(.*?)\]\((.*?)\)',part);chunks.append(f'<figure><img loading="lazy" src="{html.escape(match[2])}"><figcaption>{html.escape(match[1])}</figcaption></figure>')
  elif part.startswith('#'):
   level=len(part)-len(part.lstrip('#'));chunks.append(f'<h{level}>{html.escape(part.lstrip("# "))}</h{level}>')
  else:chunks.append('<p>'+html.escape(part)+'</p>')
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Non-stationary multidimensional Q</title><style>body{max-width:1400px;margin:auto;padding:30px;font:16px/1.7 system-ui}img{width:100%}</style>'+''.join(chunks))
 (out/'STATUS.json').write_text(json.dumps(dict(comparison_complete=len(complete),comparison_total=len(comparisons)),indent=2))
 print('Report',len(complete),'/',len(comparisons),out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
