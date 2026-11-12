from pathlib import Path
import csv,json,statistics,math
P=Path(__file__).resolve().parent;s=json.loads((P/'evidence.json').read_text());out={}
for r in s['runs']:
 seed=r['job']['seed'];m=256 if r['job']['phase']=='nm_pilot' else 64
 rows=[]
 for z in csv.DictReader(r['csv'].splitlines()):
  if not z.get('train/n_updates'):continue
  x={k:float(v) for k,v in z.items() if v and k not in ['time/time_elapsed']}
  x['steps']=12800+int(x['train/n_updates'])*8;rows.append(x)
 out[seed,m]=rows
for seed in [5,6]:
 a={r['steps']:r for r in out[seed,64]};b={r['steps']:r for r in out[seed,256]};steps=sorted(a.keys()&b.keys())
 print('\nSEED',seed,'shared',steps[0],steps[-1],len(steps))
 def first(pred,k=3):
  return next((steps[i] for i in range(len(steps)-k+1) if all(pred(t) for t in steps[i:i+k])),None)
 print('first Qdiff>10 x3',first(lambda t:abs(a[t]['train/current_q_values']-b[t]['train/current_q_values'])>10))
 print('first Q<-100 x3 M64/M256',first(lambda t:a[t]['train/current_q_values']< -100),first(lambda t:b[t]['train/current_q_values']< -100))
 print('first ESS<4 x3',first(lambda t:b[t]['train/importance_ess_mean']<4),'maxw>.5 x3',first(lambda t:b[t]['train/importance_max_weight_mean']>.5))
 for target in [13056,100000,200000,300000,400000,500000,600000,700000,800000,1000000,1200000,1500000]:
  t=min(steps,key=lambda t:abs(t-target));ra=a[t];rb=b[t]
  print(t, 'Q',round(ra['train/current_q_values'],2),round(rb['train/current_q_values'],2),'loss',round(ra['train/critic_loss'],2),round(rb['train/critic_loss'],2),'ESS',round(rb['train/importance_ess_mean'],2),'maxw',round(rb['train/importance_max_weight_mean'],3),'qstd/T',round(rb['train/proposal_q_std']/.1,2),'logrhoStd',round(rb['train/proposal_log_density_std'],2),'noise',round(ra.get('exploration/noise_std',0),3),round(rb.get('exploration/noise_std',0),3))
(P/'training-series.json').write_text(json.dumps({f'{seed}-{m}':rr for (seed,m),rr in out.items()},indent=2))
