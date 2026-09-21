from pathlib import Path
import json,csv
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parent
D=P/'runs'
summary={}; frozen=defaultdict(list); modes=defaultdict(list)
for d in D.glob('frozen*'):
 args=json.loads((d/'manifest.json').read_text())['arguments']; case=args['case']; init=args['initialization']; seed=args['seed']
 for r in csv.DictReader((d/'estimates.csv').open()):
  if int(r['k'])==50 and ((int(r['step'])==20000 and r['method'] in ('optiq_raw','optiq_td','reference')) or r['method'].startswith('local')):
   frozen[(case,init,r['method'])].append({'seed':seed,**{k:float(r[k]) for k in ('bias','rmse','mean','truth')}})
 a=np.load(d/'distribution_20000.npz'); m=a['mode_mass']; t=a['reference_mode_mass']; sample=a['samples']
 modes[(case,init)].append({'seed':seed,'tv_bins':float(abs(m-t).sum()/2),'effective_modes_actor':float(1/(m@m)),'effective_modes_target':float(1/(t@t)),'mean_action_std':float(sample.std(0).mean()),'covered_target_mass':float(t[m>0].sum())})
rows=[]
for key,rr in sorted(frozen.items()):
 rows.append(dict(case=key[0],initialization=key[1],method=key[2],seeds=','.join(str(r['seed']) for r in sorted(rr,key=lambda r:r['seed'])),n=len(rr),**{k+'_mean':float(np.mean([r[k] for r in rr])) for k in ('bias','rmse')},rmse_seed_sd=float(np.std([r['rmse'] for r in rr],ddof=1)) if len(rr)>1 else 0.))
with (P/'frozen_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader();w.writerows(rows)
summary['frozen']=[r for r in rows if r['case'] in ('unimodal','asymmetric_1.3_0.08','modes2d_4','modes2d_8','separable_4','separable_8') and (r['method'] in ('optiq_raw','optiq_td') or r['method'].startswith('local_no_is'))]
summary['modes']=[dict(case=k[0],initialization=k[1],data=v) for k,v in modes.items() if k[0] in ('modes2d_4','modes2d_8','separable_4','separable_8')]
summary['movecar']=[]
for d in sorted(D.glob('movecar*')):
 args=json.loads((d/'manifest.json').read_text())['arguments']; rows=json.loads((d/'learning.json').read_text()); final=rows[-1]
 times=[r['step'] for r in rows if r['return_mean']>=180]
 summary['movecar'].append(dict(method=args['method'],seed=args['seed'],first_observed_180=min(times) if times else None,**final))
summary['controls']=[]
for d in sorted(D.glob('control*')):
 a=np.load(d/'grid_convergence.npz'); b=np.load(d/'landscape_20000.npz'); rows=json.loads((d/'training.json').read_text())
 summary['controls'].append(dict(name=d.name,grid_error=float(a['max_error']),final=rows[-1],barrier=float(b['path'].max()-max(b['path'][0],b['path'][-1])),down_fraction=float((b['perturbation'].min(1)<-1e-6).mean())))
summary['landscape']=[dict(name=d.name,metrics=json.loads((d/'metrics.json').read_text())) for d in sorted(D.glob('landscape*'))]
summary['learned']=[]
for d in sorted(D.glob('learned*')):
 rows=list(csv.DictReader((d/'learned_backup.csv').open()))
 for method in sorted({r['method'] for r in rows}):
  selected=[r for r in rows if r['method']==method and int(r['k'])==50]
  summary['learned'].append(dict(name=d.name,method=method,states=len(selected),mean_bias=float(np.mean([float(r['bias']) for r in selected])),mean_rmse=float(np.mean([float(r['rmse']) for r in selected]))))
(P/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
print('MOVE',json.dumps(summary['movecar']))
print('CONTROL',json.dumps(summary['controls']))
print('LEARNED',json.dumps(summary['learned']))
for row in summary['frozen']:
 if row['method'] in ('optiq_raw','optiq_td'): print('FROZEN',row)
for row in summary['modes']: print('MODE',row)
