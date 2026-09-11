from pathlib import Path
import json,csv,importlib.util
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parent;D=P/'runs'
spec=importlib.util.spec_from_file_location('problems',P/'source/problems.py');problems=importlib.util.module_from_spec(spec);import sys;sys.modules['problems']=problems;spec.loader.exec_module(problems)
allrows=[];modes=[];local_support=[]
for d in sorted(D.glob('frozen*')):
 a=json.loads((d/'manifest.json').read_text())['arguments'];assert a['seed']<5
 allrows.extend([{**r,'seed':a['seed'],'case':a['case'],'initialization':a['initialization']} for r in csv.DictReader((d/'estimates.csv').open())])
 for f in sorted(d.glob('distribution_*.npz')):
  step=int(f.stem.rsplit('_',1)[1]);x=np.load(f);m=x['mode_mass'];t=x['reference_mode_mass'];samples=x['samples'];modes.append(dict(case=a['case'],initialization=a['initialization'],seed=a['seed'],step=step,bin_tv=float(abs(m-t).sum()/2),effective_modes=float(1/(m@m)),covered_target_mass=float(t[m>0].sum()),coord_std=float(samples.std(0).mean())))
 if a['seed']==0 and a['initialization']=='default':
  p=problems.make_problem(a['case']);grid,mass,truth,_,_=p.truth()
  centers=[np.repeat(c[0],p.dim) for c in p.centers] if p.separable else p.centers
  for i,center in enumerate(centers):
   if p.separable:
    mask=np.abs(grid[:,0]-center[0])<=.5;prob=float(mass[mask].sum())**p.dim;conditional=float(mass[mask]@p.base().q(grid[mask])/mass[mask].sum())*p.dim
   else:
    mask=np.all(np.abs(grid-center)<=.5,axis=1);prob=float(mass[mask].sum());conditional=float(mass[mask]@p.q(grid[mask])/prob)
   local_support.append(dict(case=a['case'],center=i,target_support_mass=prob,conditional_truth=conditional,global_truth=truth,bias_limit=conditional-truth))
# Bias versus variance identity, with sample standard deviations corrected to n denominator.
identity=max(abs(float(r['rmse'])**2-float(r['bias'])**2-(1-1/int(r['repetitions']))*float(r['std'])**2) for r in allrows)
learned=[]
for d in sorted(D.glob('learned*')):
 seed=int(d.name.split('seed')[-1]);rr=list(csv.DictReader((d/'learned_backup.csv').open()));assert seed<5
 for k in [1,50,256]:
  by=defaultdict(dict)
  for r in rr:
   if int(r['k'])==k:by[float(r['state'])][r['method']]=r
  for st,r in by.items():
   raw=r['optiq_raw'];td=r['optiq_td'];local=r['local_no_is']
   learned.append(dict(seed=seed,k=k,state=st,truth=float(raw['truth']),raw_bias=float(raw['bias']),td_bias=float(td['bias']),local_bias=float(local['bias']),raw_rmse=float(raw['rmse']),td_rmse=float(td['rmse']),local_rmse=float(local['rmse']),noise_shift=float(td['mean'])-float(raw['mean'])))
paths=[]
for d in sorted(D.glob('cross*')):
 a=np.load(d/'landscape.npz');seed=int(d.name.split('seed')[-1]);assert seed<5
 for key in a.files:
  if not key.endswith('_path'):continue
  critic=key[:-5];v=a[key];diff=np.diff(v);extent=np.ptp(v)
  paths.append(dict(run=d.name,seed=seed,critic=critic,start=float(v[0]),end=float(v[-1]),drop=float(v[0]-v[-1]),barrier=float(v.max()-max(v[0],v[-1])),variation_ratio=float(abs(diff).sum()/max(extent,1e-12)),reversal_steps=int(np.sum(diff>1e-4)),end_r05_median_abs_delta=float(np.median(abs(a[critic+'_end_r0.05']))),end_r05_second_difference=float(np.median(abs(a[critic+'_end_r0.05'].sum(1))))))
controls=[]
for d in sorted(D.glob('control*')):
 seed=int(d.name.split('seed')[-1]);arm=d.name.split('_')[1];a=np.load(d/'landscape_20000.npz');v=a['path'];prec=json.loads((d/'reference_precision.json').read_text());train=json.loads((d/'training.json').read_text());steps=[]
 for f in d.glob('landscape_*.npz'):
  b=np.load(f);vv=b['path'];steps.append(dict(step=int(f.stem.split('_')[-1]),barrier=float(vv.max()-max(vv[0],vv[-1]))))
 controls.append(dict(arm=arm,seed=seed,barrier=float(v.max()-max(v[0],v[-1])),variation_ratio=float(abs(np.diff(v)).sum()/max(np.ptp(v),1e-12)),drop=float(v[0]-v[-1]),perturbation_second_difference=float(np.median(abs(a['perturbation'].sum(1)))),grid_error=prec['coarse_to_reference'],reference_n=prec['reference_n'],critic_loss_last1000=float(np.mean([r['critic_loss'] for r in train if r['step']>19000])),barrier_steps=steps))
res=dict(frozen=allrows,mode_history=modes,local_support=local_support,variance_identity_error=identity,learned=learned,cross_paths=paths,controls=controls)
(P/'deep_analysis.json').write_text(json.dumps(res,indent=2))
for key in ['mode_history','local_support','learned','cross_paths','controls']:
 data=res[key];keys=[k for k,v in data[0].items() if not isinstance(v,list)]
 with (P/(key+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(data)
print('VARIANCE_IDENTITY',identity)
for arm in ['max','boltzmann','actor','local']:
 r=[x for x in controls if x['arm']==arm];print('CONTROL',arm,{k:np.mean([x[k] for x in r]) for k in ['barrier','variation_ratio','drop','perturbation_second_difference','critic_loss_last1000']},'barriers',[x['barrier'] for x in r])
for family in ['cross_ddpg_sd2','cross_td3_sd3','cross_optiq_optiq']:
 for critic in sorted({x['critic'] for x in paths if x['run'].startswith(family)}):
  r=[x for x in paths if x['run'].startswith(family) and x['critic']==critic];print('CROSS',family,critic,'barriers',[round(x['barrier'],4) for x in r],'ratios',[round(x['variation_ratio'],3) for x in r],'drops',[round(x['drop'],3) for x in r])
for seed in range(5):
 r=[x for x in learned if x['seed']==seed and x['k']==50];print('LEARNED',seed,'bias',*[np.mean([x[k] for x in r]) for k in ['raw_bias','td_bias','local_bias','noise_shift']],'raw_win_states',sum(x['raw_rmse']<x['local_rmse'] for x in r),'negative_noise',sum(x['noise_shift']<0 for x in r))
