"""Read-only checkpoint validation, fresh random latent keys, original metric."""
import argparse,os,json,sys,hashlib,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--step',type=int,default=100000);a=p.parse_args()
c=json.loads((a.run/'config.json').read_text());assert c['method']=='optiq_trg' and c['latent_mode']=='random'
source=Path('/home/heechan/OptiQ-ops/sources')/c['source_git_commit'];sys.path.insert(0,str(source));os.environ.update(GMM40_REPO_ROOT=str(source),GMM40_RESULTS_ROOT=str(a.run.parent),JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
from gmm40.optiq_trg import OptiQTRG
from gmm40.target import Target
from gmm40.evaluation import assignments,metrics
from gmm40.mu90_launch import unchanged_algorithm
unchanged_algorithm(source)
t=Target();agent=OptiQTRG(t,seed=c['seed'],n=c['n'],m=c['m'],batch=c['batch'],hidden_dims=(c['width'],)*c['depth'],temperature=c['temperature'],log_std_max=c['trg_log_std_max'],initial_log_std=c['trg_initial_log_std'],teacher_std_floor=c['trg_teacher_std_floor'],mean_output_init_scale=c['mean_output_init_scale'])
ckpt=a.run/'checkpoints'/f'step_{a.step:07d}.bin';agent.restore(ckpt);assert agent.updates==a.step
n=10000;ref=t.sample(n,20260917,bounded=True);full=t.sample(n,20260917,bounded=False);rows=[];keys=[900000+c['seed']]+list(range(2026092101,2026092106));a.out.mkdir(parents=True,exist_ok=False)
for key in keys:
 _,_,extra=agent.evaluate_samples(n,key);x=extra['mu_only'];m=metrics(x,t,ref,full);np.save(a.out/f'mu-key{key}.npy',x);rows.append(dict(latent_key=key,**m));print(key,m['high_density_fraction'],m['mode_coverage'],flush=True)
original=json.loads((a.run/'evaluations'/f'step_{a.step:07d}'/'metrics_mu_only.json').read_text())
assert abs(rows[0]['high_density_fraction']-original['high_density_fraction'])<.005
assert rows[0]['mode_coverage']==original['mode_coverage']
# No target filtering and no sigma added. All five held-out sets count.
result=dict(run=str(a.run),source_commit=c['source_git_commit'],resume_runner_commit=c.get('resume_runner_commit'),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),evaluation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend='CPU',training_updates=agent.updates,original_reproduction=rows[0],independent=rows[1:],all_five_pass=all(x['mode_coverage']==40 and x['high_density_fraction']>=.9 for x in rows[1:]),original_gpu_near=original['high_density_fraction'],created=time.time())
(a.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
