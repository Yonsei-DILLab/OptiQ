"""Check all resolved configs and actual initialized sigma before training."""
import copy
import json
import sys
import numpy as np
import jax
import jax.numpy as jnp
import train_sigma as abla
from omegaconf import OmegaConf
from optiq_dime.policy import SemiImplicitActor

rows=[]
for task in ('ant','walker2d','humanoid','hopper','halfcheetah'):
    reference=None
    for sigma in abla.SIGMAS:
        sys.argv=['preflight',str(sigma)]
        cfg=abla.compose_config([f'benchmark={task}'])
        algorithm=copy.deepcopy(OmegaConf.to_container(cfg.alg,resolve=True))
        algorithm['actor'].pop('initial_log_std')
        if reference is None:reference=algorithm
        assert algorithm==reference, 'Only initial scale may differ'
        rows.append(dict(task=task,sigma=sigma,log_sigma=cfg.alg.actor.initial_log_std))
for sigma in abla.SIGMAS:
    actor=SemiImplicitActor(action_dim=8,hidden_dims=(256,256),log_std_min=-5,
        log_std_max=0,initial_log_std=float(np.log(sigma)))
    obs=jnp.ones((4,27));z=jax.random.normal(jax.random.PRNGKey(0),(4,8))
    params=actor.init(jax.random.PRNGKey(1),obs,z)
    mu,log_std=actor.apply(params,obs,z)
    np.testing.assert_allclose(np.exp(np.asarray(log_std)),sigma,rtol=1e-6)
assert '20260920_truncated_mll' in sys.modules['optiq_dime.policy'].__file__
print(json.dumps(dict(passed=True,configs=rows,implementation=sys.modules['optiq_dime.policy'].__file__)))

# Exercise the actual JIT loss metrics, including values above the old -1 cap.
import tempfile
from optiq_dime.algorithm import OptiQDIME
sys.argv=['preflight','0.1']
cfg=abla.compose_config(['benchmark=ant','alg.buffer_size=256','require_gpu=false',
    'output_root='+tempfile.mkdtemp(prefix='sigma-max0-preflight-')])
model,callbacks=abla.base.runner.create_algorithm(cfg)
try:
    a=cfg.alg.actor;c=cfg.alg.critic
    kwargs={k:a[k] for k in ('num_policy_samples','proposals_per_policy_sample',
        'proposal_sampling_mode','proposal_std','proposal_clip','include_anchor',
        'density_correction','density_beta','adaptive_density_beta','minimum_source_ess',
        'density_beta_grid_size','temperature','sinkhorn_epsilon','sinkhorn_iterations',
        'source_q_eval','transport_target_mode')}
    kwargs.update(semi_implicit=True,normalize_ot_cost=False,distillation_loss='direct_gmm_nll',
        teacher_distribution='conditional_mixture',entropy_diagnostics=False,ot_student_action='mean')
    for bias,expected in ((-.5,0.),(.5,1.)):
        state=model.policy.actor_state
        params=copy.deepcopy(state.params)
        params['log_std']['bias']=jnp.full_like(params['log_std']['bias'],bias)
        state=state.replace(params=params)
        _,_,_,metrics=OptiQDIME.update_actor(state,model.policy.qf_state,
            jnp.zeros((2,*model.observation_space.shape)),jax.random.PRNGKey(7),
            jnp.linspace(c.v_min,c.v_max,c.n_atoms),**kwargs)
        assert float(metrics['actor_std_at_max_fraction'])==expected
        assert np.isfinite(float(metrics['actor_loss']))
        print(json.dumps(dict(raw_log_sigma=bias,expected_at_max_fraction=expected,
            actual_at_max_fraction=float(metrics['actor_std_at_max_fraction']))))
finally:
    model.get_env().close();callbacks.callbacks[0].eval_env.close();model.logger.close()
