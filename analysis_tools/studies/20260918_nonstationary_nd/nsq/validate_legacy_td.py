"""Independent zero-noise TD check plus actual legacy actor/critic continuation."""
import argparse,json,time,traceback
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from .models import actor_state,critic_state,td_update,sample_action
from .run import run
from .io import verify_source,sha
from .config import write_json

def distance(a,b):
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))

def main(root,dim):
    root=Path(root);code=verify_source(root);assert jax.default_backend()=='gpu'
    rec=dict(passed=False,source_code_id=code,commit=json.loads((root/'DEPLOYMENT.json').read_text())['commit'],dim=dim,gpu=str(jax.devices()[0]),checks=[])
    dest=root/f'LEGACY_TD_VALIDATION_D{dim}.json'
    try:
        actor=actor_state('argmax_truncated',0,dim);critic=critic_state(0,dim);key=jax.random.PRNGKey(919)
        rng=np.random.default_rng(18)
        batch={k:jnp.asarray(rng.uniform(-1,1,(256,dim)).astype(np.float32)) for k in ['s','a','sp']};batch['r']=jnp.asarray(rng.normal(size=256).astype(np.float32))
        keys=jax.random.split(key,6);ap=sample_action(actor,batch['sp'],keys[1],deterministic=False)
        tq=critic.apply_fn({'params':critic.target_params,'batch_stats':critic.target_batch_stats},batch['sp'],ap,train=False)
        target=jax.lax.stop_gradient(batch['r']+.99*tq[...,0].min(0))
        def objective(params):
            q=critic.apply_fn({'params':params,'batch_stats':critic.batch_stats},batch['s'],batch['a'],train=False)[...,0]
            return jnp.square(q-target[None]).mean(1).sum()
        ref_loss,grad=jax.value_and_grad(objective)(critic.params)
        expected=critic.apply_gradients(grads=grad)
        expected=expected.replace(target_params=jax.tree_util.tree_map(lambda p,q:.005*p+.995*q,expected.params,critic.target_params))
        actual,metrics,newkey=td_update(actor,critic,batch,key)
        jax.block_until_ready(actual.params)
        gaps=dict(loss=abs(float(metrics['critic_loss'])-float(ref_loss)),target=abs(float(metrics['next_q_values'])-float(target.mean())),parameters=distance(actual.params,expected.params),target_parameters=distance(actual.target_params,expected.target_params))
        assert all(np.isfinite(v) and v<2e-5 for v in gaps.values()),gaps
        np.testing.assert_array_equal(newkey,keys[0]);rec['independent_td_gaps']=gaps;rec['checks'].append('TD target, twin MSE, Adam gradient update, target EMA and RNG match independent noiseless calculation')
        task=dict(name=f'legacy_closed_D{dim}_continuous',stage='closed',family='tri',dim=dim,method='argmax_truncated',n=16,m=64,seed=0,updates=32,parent=None,q_source=None,actor_batch=1,priority=0)
        run(root,task,validation=True,max_updates=32,warmup=32,skip_evaluation=True)
        resumed=dict(task,name=f'legacy_closed_D{dim}_resumed')
        try:run(root,resumed,validation=True,max_updates=32,warmup=32,interrupt_at=16,skip_evaluation=True)
        except SystemExit as e:assert e.code==75
        else:raise AssertionError('Expected checkpoint interruption')
        run(root,resumed,validation=True,max_updates=32,warmup=32,skip_evaluation=True)
        paths=[root/'validation_runs'/t['name']/'checkpoint.msgpack' for t in [task,resumed]]
        x,y=[serialization.msgpack_restore(p.read_bytes()) for p in paths]
        ck_gaps={k:distance(x[k],y[k]) for k in ['actor','critic','key','collectkey','env_state']}
        assert max(ck_gaps.values())<2e-5,ck_gaps
        for k in x['replay']['data']:np.testing.assert_allclose(x['replay']['data'][k],y['replay']['data'][k],atol=2e-6)
        assert x['replay']['rng']==y['replay']['rng']
        rec['checkpoint_gaps']=ck_gaps;rec['checkpoint_sha256']=[sha(p) for p in paths];rec['checks'].append('32 actual legacy actor/critic updates finite; interrupted16+16 agrees with uninterrupted32 including replay and RNG')
        rec['passed']=True
    except BaseException:
        rec['error']=traceback.format_exc();raise
    finally:
        rec['time']=time.time();write_json(dest,rec);print(json.dumps(rec),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--dim',type=int,required=True);a=p.parse_args();main(a.root,a.dim)
