"""Read-only postprocessing; never imports or updates the training code."""
from pathlib import Path
import argparse,json,time,collections,concurrent.futures,shutil
import numpy as np

KEYS=['step','first_axis_tv','mean_marginal_tv','max_marginal_tv','basin_mass_tv','sliced_w1','backup_bias','sigma_mean','between_mu_variance','within_variance','teacher_first_tv','teacher_basin_mass_tv','actor_teacher_tv','probe_ess','probe_wmax','usage_ess','underused_fraction','row_entropy','reference_ess','reference_reliable','reference_half_tv','stochastic_return_mean','stochastic_return_sd','train_cumulative','elapsed_seconds','joint_2d_tv']
DKEYS=['step','edges','marginal_mass','target_marginals','proposal_marginals','teacher_marginals','basin_mass','target_basin_mass','boundaries','grid','q_slice','joint_edges','joint_mass','target_joint_mass','fixed_mu','fixed_log_sigma','reference_reliable','reference_ess','reference_half_tv']
TIMES=[0,1,1000,10000,20000,20001,20020,20200,21000,22000,25000,25001,25020,25200,26000,27000,30000,30001,30020,30200,35000]

def main(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 inventory=json.loads((root/'scope_reduction_20260919/FROZEN_INVENTORY.json').read_text())
 for t in inventory['tasks']:
  p=Path(t['root'])/'runs'/t['name']/'RUNNING.json'
  t['host']=json.loads(p.read_text()).get('host','unknown') if p.exists() else 'unknown'
 (out/'inventory.json').write_text(json.dumps(inventory,indent=2))
 selected=[t for t in inventory['tasks'] if t['status']=='complete' and t['stage']!='source']
 def one(t):
  src=Path(t['root'])/'runs'/t['name'];dst=out/'runs'/t['name'];dst.mkdir(parents=True,exist_ok=True)
  if (dst/'DONE.json').exists():return t['name']
  stats={};evaluations=[]
  # Prefix time series do not enter post-change AUC; retain exact final histogram.
  if t['stage']!='prefix':
   for p in sorted((src/'evaluations').glob('*.npz')):
    with np.load(p) as z:
     evaluations.append({k:z[k] for k in KEYS if k in z})
   if evaluations:
    allkeys=set().union(*(r.keys() for r in evaluations));data={k:np.concatenate([r.get(k,np.full(len(r['step']),np.nan)) for r in evaluations]) for k in allkeys}
    _, revix=np.unique(data['step'][::-1],return_index=True);ix=np.sort(len(data['step'])-1-revix);ix=ix[np.argsort(data['step'][ix])];data={k:v[ix] for k,v in data.items()};np.savez_compressed(dst/'series.npz',**data)
  # The last 200 actual training updates quantify numerical OT convergence.
  files=sorted((src/'metrics').glob('*.npz'))
  if files:
   with np.load(files[-1]) as z:
    for k in ['ot_row_l1','ot_col_l1','effective_col_l1','zero_rows','ess','wmax','gradient_norm','cost_mean','sigma_min','sigma_max']:
     if k in z:stats[k]=float(np.mean(z[k]))
  (dst/'stats.json').write_text(json.dumps(stats))
  for name in ['COMPLETE.json','progress.json','config.json','RUNNING.json']:
   if (src/name).exists():shutil.copy2(src/name,dst/name)
  snaps=['final_independent']+([f'{x:06d}' for x in TIMES] if t['seed']==0 else [])
  for name in snaps:
   p=src/'density'/(name+'.npz')
   if p.exists():
    with np.load(p) as z:data={k:z[k] for k in DKEYS+KEYS if k in z}
    (dst/'density').mkdir(exist_ok=True);np.savez_compressed(dst/'density'/(name+'.npz'),**data)
  if t['seed']==0 and t['n']==16:
   for name in ['020001','025001','final_independent']:
    p=src/'assignments'/(name+'.npz')
    if p.exists():(dst/'assignments').mkdir(exist_ok=True);shutil.copy2(p,dst/'assignments'/p.name)
  (dst/'DONE.json').write_text(json.dumps(dict(source=src.as_posix(),completed=time.time())))
  return t['name']
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for i,name in enumerate(pool.map(one,selected),1):
   if i%100==0 or i==len(selected):print('COLLECTED',i,'/',len(selected),name,flush=True)
 print('REPORT_INPUT_READY',out,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.root,a.out)
