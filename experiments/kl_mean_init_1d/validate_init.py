"""Paired initialization isolation and measured latent mean diversity."""
import argparse,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from experiments.kl_direction_1d.core import Experiment

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 base=json.loads(Path('experiments/kl_direction_1d/config.json').read_text());new=json.loads(Path(__file__).with_name('config.json').read_text())
 training_keys=['n','m','batch','steps','temperature','learning_rate','hidden_dims','log_std_min','log_std_max','initial_log_std','teacher_std_floor']
 for k in training_keys:assert base[k]==new[k],k
 z=jax.random.normal(jax.random.PRNGKey(92731),(4096,1));records=[]
 for seed in new['seeds']:
  a=Experiment(base,'forward',0,seed);b=Experiment(new,'forward',0,seed)
  pa,pb=flatten_dict(a.state.params),flatten_dict(b.state.params)
  assert pa.keys()==pb.keys()
  for key in pa:
   if key!=('mu','kernel'):assert np.array_equal(np.asarray(pa[key]),np.asarray(pb[key])),key
  ratio=float(np.linalg.norm(np.asarray(pb['mu','kernel']))/np.linalg.norm(np.asarray(pa['mu','kernel'])))
  assert np.isclose(ratio,100,rtol=1e-6)
  relative=float(np.linalg.norm(np.asarray(pb['mu','kernel'])-100*np.asarray(pa['mu','kernel']))/np.linalg.norm(np.asarray(pb['mu','kernel'])))
  assert relative<1e-6
  mu0,ls0=a.components(a.state.params,z);mu1,ls1=b.components(b.state.params,z)
  assert np.array_equal(np.asarray(ls0),np.asarray(ls1))
  records.append(dict(seed=seed,other_parameters_identical=True,mean_kernel_norm_ratio=ratio,scaled_kernel_relative_error=relative,
    initial_mean_std_old=float(mu0.std()),initial_mean_std_new=float(mu1.std()),initial_mean_range_old=[float(mu0.min()),float(mu0.max())],initial_mean_range_new=[float(mu1.min()),float(mu1.max())]))
 data=dict(passed=True,records=records);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
if __name__=='__main__':main()
