import argparse,json,time,tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import quad
from .core import Experiment
from .target import np_pdf,reference,geometry

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',action='store_true');args=p.parse_args()
 cfg=json.loads(Path(__file__).with_name('config.json').read_text());checks={}
 for case in cfg['cases']:
  c=dict(cfg,**case);e=Experiment(dict(c,n=8,m=11,batch=2,hidden_dims=[16,16]),'reverse',1024,0)
  a=np.linspace(-9.9,9.9,137);score=np.asarray(e.target_score(jnp.asarray(a[:,None])))[:,0]
  h=1e-5;fd=(np.log(np_pdf(c,a+h))-np.log(np_pdf(c,a-h)))/(2*h)
  err=float(np.max(abs(score-fd)));assert err<2e-4,(case['id'],err)
  dens=np.exp(np.asarray(e.log_f(jnp.asarray(a[:,None]))));assert np.max(abs(dens-np_pdf(c,a)))<1e-5
  integral=quad(lambda x:float(np_pdf(c,x)),-10,10,epsabs=1e-10,limit=500)[0];assert abs(integral-1)<1e-8
  edges=np.linspace(-10,10,513);cdf=reference(c,edges)[1]
  # Independent adaptive integration for every histogram bin, including steep edges.
  bins=np.array([quad(lambda x:float(np_pdf(c,x)),l,r,epsabs=1e-12)[0] for l,r in zip(edges[:-1],edges[1:])]);cdferr=float(np.max(abs(np.diff(cdf)-bins)));assert cdferr<2e-7,cdferr
  geom=geometry(c);assert np.all(geom['target_core']>0)
  checks[case['id']]=dict(score_abs_error=err,normalization_error=abs(integral-1),bin_mass_max_error=cdferr,peaks=geom['peaks'].tolist())
 with tempfile.TemporaryDirectory() as tmp:
  e.advance(2);e.save(Path(tmp)/'c');e.advance(2);expected=jax.tree_util.tree_leaves(e.state.params)
  e.restore(Path(tmp)/'c');e.samples(128);e.advance(2)
  assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(expected,jax.tree_util.tree_leaves(e.state.params)))
 timings=[]
 if args.gpu:
  assert jax.default_backend()=='gpu'
  for method,L in [('forward',0),('reverse',1024)]:
   jax.clear_caches();e=Experiment(dict(cfg,**cfg['cases'][0]),method,L,0);e.advance(100);t=time.perf_counter();info=e.advance(100)
   assert all(np.isfinite(v) for v in info.values());timings.append(dict(method=method,seconds_per_update=(time.perf_counter()-t)/100))
 result=dict(passed=True,backend=jax.default_backend(),checks=checks,resume_rng_exact=True,timing=timings);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
