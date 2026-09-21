import importlib.util,json,sys,subprocess,types
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
root=Path(__file__).resolve().parents[3];sys.path[:0]=[str(root),str(root/'tests')]
from optiq_dime import OptiQDIME
from test_semi_implicit import actor_state,critic_state
from optiq_dime.optimizers import adam_with_grad_clip
commit='8cb4f237aacb61113f65e693e23236f17d77f978'
def historical(module):
    name='optiq_dime._release_old_'+module
    code=subprocess.check_output(['git','show',commit+':optiq_dime/'+module+'.py'],cwd=root,text=True)
    mod=types.ModuleType(name);mod.__package__='optiq_dime';sys.modules[name]=mod
    exec(compile(code,commit+':'+module,'exec'),mod.__dict__)
    return mod
old=historical('algorithm').OptiQDIME
old_guard=historical('soft_improvement')
from optiq_dime.soft_improvement import sampled_soft_update
actor,critic=actor_state(),critic_state()
tx=adam_with_grad_clip(3e-4,.9,.999,2.)
actor=actor.replace(tx=tx,opt_state=tx.init(actor.params))
critic=critic.replace(tx=tx,opt_state=tx.init(critic.params))
obs=jnp.ones((4,3));key=jax.random.PRNGKey(53)
a_args=(actor,critic,obs,key,jnp.array([-3600.]),16,4,'exact',.05,.5,False,True,1.,False,16.,257,.1,.25,100,'mean','argmax',True,False,'conditional_ot_nll','conditional_mixture')
a_old=old.update_actor(*a_args);a_new=OptiQDIME.update_actor(*a_args)
c_args=(False,False,.99,actor,critic,obs,jnp.zeros((4,2)),obs,jnp.arange(4,dtype=jnp.float32),jnp.array([0.,1.,0.,0.]),1,jnp.array([-3600.]),-3600.,3600.,0.,0.,0.,key,True,16,.1)
c_old=old.update_critic(*c_args);c_new=OptiQDIME.update_critic(*c_args)
g_args=(actor,a_new[0],critic,jnp.ones((32,3)),key,.1,jnp.array([-3600.]),16,8,0.)
g_old=old_guard.sampled_soft_update(*g_args);g_new=sampled_soft_update(*g_args)
checks={}
for label,x,y in [('actor_trainstate',a_old[0],a_new[0]),('actor_loss',a_old[1],a_new[1]),('actor_rng',a_old[2],a_new[2]),('critic_trainstate',c_old[0],c_new[0]),('critic_metrics',c_old[1],c_new[1]),('critic_rng',c_old[2],c_new[2]),('guard_trainstate',g_old[0],g_new[0]),('guard_metrics',g_old[1],g_new[1])]:
    extra=[]
    if isinstance(x,dict):
        assert set(x) <= set(y),label
        extra=sorted(set(y)-set(x));y={k:y[k] for k in x}
    a,b=jax.tree_util.tree_leaves(x),jax.tree_util.tree_leaves(y)
    assert len(a)==len(b),label
    for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb,err_msg=label)
    checks[label]={'array_equal':True,'array_count':len(a),'additional_diagnostics':extra}
report={'training_commit':commit,'device':str(jax.devices()),'scope':'Original algorithm/guard function bodies vs current common implementation, same initialized states/base draws, final continuous settings; shared unchanged numerical helpers. Four synthetic states and small-network fixtures, production Adam/clipping parameters, full N16/K64/M16/J8/BG32. Not a long-run performance replication.','checks':checks}
(root/'docs/v2/review').mkdir(exist_ok=True)
(root/'docs/v2/review/historical_update_parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
