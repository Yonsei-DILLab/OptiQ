"""Validate gradients/paired initialization; optionally benchmark full L on GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from .core import Experiment, target_score, log_f, q_value, dense_score
from ..kl_forward_wide_1d.core import Experiment as Forward
from ..kl_forward_wide_1d.box_gaussian import sample_box, mixture_log_prob


def flat(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(tree)])


def close(a, b, tol=3e-5):
    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert error < tol*max(1., float(np.max(np.abs(b)))), error
    return error


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--out',type=Path)
    args=p.parse_args();cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    prior=json.loads((Path(__file__).parent.parent/'kl_forward_wide_1d/config.json').read_text())
    report={'checks':{},'backend':jax.default_backend(),'devices':[str(x) for x in jax.devices()]}
    checks=report['checks']
    assert cfg['n']==cfg['m']==128 and cfg['steps']==100000 and cfg['conditions']==[{'method':'reverse','L':1048576}]
    changed={k for k in set(cfg)|set(prior) if cfg.get(k)!=prior.get(k)}
    assert changed=={'study','conditions','density_chunk','train_block','checkpoint_interval'}, changed
    checks['config_difference']=sorted(changed)
    paired=[]
    for seed in cfg['seeds']:
        e=Experiment(cfg,'reverse',1048576,seed);f=Forward(prior,'forward',0,seed)
        left=flax.serialization.to_bytes({'state':e.state,'key':e.key})
        assert left==flax.serialization.to_bytes({'state':f.state,'key':f.key})
        assert all(np.array_equal(x,y) for x,y in zip(e.samples(1024),f.samples(1024)))
        paired.append({'seed':seed,'state_sha256':hashlib.sha256(left).hexdigest(),'samples_identical':True})
    checks['matched_initialization']=paired
    small=dict(cfg,n=8,m=11,batch=2,hidden_dims=[16,16],density_chunk=8)
    e=Experiment(small,'reverse',32,0);key=jax.random.PRNGKey(701)
    sk,ik,ek,dk=jax.random.split(key,4)
    z=jax.random.normal(sk,(8,1));idx=jax.random.randint(ik,(11,),0,8)
    aux=jnp.concatenate([jax.random.normal(jax.random.fold_in(dk,i),(8,1)) for i in range(4)])
    a=jnp.linspace(-9.7,9.7,31)[:,None]
    checks['target_gradient']=close(target_score(a),jax.grad(lambda x:log_f(x).sum())(a))
    mu,ls=e.components(e.state.params,aux)
    s,lp,_=e.density_score(e.state.params,a,dk,32);s0,lp0=dense_score(a,mu,ls)
    checks['chunk_score_vs_dense']=close(s,s0)
    checks['chunk_log_density_vs_dense']=close(lp,lp0)
    old_s,old_lp,_=Forward.density_score(e,e.state.params,a,dk,32)
    checks['prior_recurrence_vs_dense_max_abs_error']=float(np.max(np.abs(np.asarray(old_s-s0))))
    checks['score_reference_max_abs']=float(np.max(np.abs(np.asarray(s0))))
    checks['score_vs_action_autograd']=close(s0,jax.grad(lambda x:mixture_log_prob(x[None],mu[None],ls[None]).sum())(a))
    def reference(params):
        cm,cl=e.components(params,z);actions=sample_box(ek,cm[idx],cl[idx])
        am,al=e.components(jax.tree_util.tree_map(jax.lax.stop_gradient,params),aux)
        return (mixture_log_prob(actions[None],am[None],al[None])[0]-q_value(actions)/cfg['temperature']).mean()
    grad=jax.grad(lambda params:e.group_loss(params,key)[0])(e.state.params)
    checks['surrogate_gradient_vs_direct_detached_density']=close(flat(grad),flat(jax.grad(reference)(e.state.params)),1e-4)
    assert np.linalg.norm(flat(grad))>1e-5
    checks['auxiliary_parameter_gradient_zero']=close(flat(jax.grad(lambda params:e.density_score(params,a,dk,32)[1].sum())(e.state.params)),0.)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state';e.advance(2);e.save(path);e.advance(2)
        expected=flat(e.state.params);expected_key=np.asarray(e.key)
        e.restore(path);e.samples(128);e.advance(2)
        checks['resume_and_eval_rng_isolation']=close(flat(e.state.params),expected,1e-7)
        assert np.array_equal(np.asarray(e.key),expected_key)
    if args.gpu:
        assert jax.default_backend()=='gpu'
        del e; jax.clear_caches()
        e=Experiment(cfg,'reverse',1048576,0)
        print(json.dumps({'phase':'compile_full_shape','L':e.L,'batch':e.batch,'chunk':cfg['density_chunk']}),flush=True)
        started=time.perf_counter();executable=e.advance_fn.lower(e.state,e.key,1).compile()
        report['compile_seconds']=time.perf_counter()-started
        report['compiled_memory_bytes']=str(executable.memory_analysis())
        measurements=[]
        for i in range(3):
            started=time.perf_counter();info=e.advance(1);seconds=time.perf_counter()-started
            assert all(np.isfinite(x) for x in info.values()) and info['gradient_norm']>0
            assert info['sigma_min']>=np.exp(-5)-1e-6 and info['sigma_max']<=np.exp(-1)+1e-6
            measurements.append({'update':i+1,'seconds':seconds,'metrics':info})
            print(json.dumps(measurements[-1]),flush=True)
        report['full_L_benchmark']=measurements
        report['estimated_training_hours_per_seed']=np.median([r['seconds'] for r in measurements[1:]])*cfg['steps']/3600
        report['device_memory_stats']=jax.devices()[0].memory_stats()
    report['passed']=True
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
