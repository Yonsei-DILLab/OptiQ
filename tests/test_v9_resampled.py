"""v9 regression tests: resampling, conditional NLL, and release profiles."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from optiq_dime.resampled_ot import multinomial_resample, resampled_transport
from optiq_dime.transport import sinkhorn
from optiq_dime.distillation import conditional_ot_nll

def test_resampling_and_duplicate_equivalence():
    key=jax.random.PRNGKey(0)
    w=jax.nn.softmax(jnp.linspace(-4, 3, 64))
    indices=jax.jit(lambda k: multinomial_resample(k,jnp.broadcast_to(w,(20000,64)),16))(key)
    counts=np.bincount(np.asarray(indices).ravel(),minlength=64)/indices.size
    assert np.max(np.abs(counts-np.asarray(w)))<.002
    flat=multinomial_resample(key,jnp.ones((8,16))/16,16)
    assert flat.shape==(8,16) and int(flat.min())>=0 and int(flat.max())<16
    atom=multinomial_resample(key,jax.nn.one_hot(jnp.array([0,63]),64),16)
    np.testing.assert_array_equal(atom,np.array([[0]*16,[63]*16]))

    mu=jax.random.normal(key,(3,16,2))*.4
    u=jax.random.normal(jax.random.fold_in(key,1),(3,64,2))
    weights=jnp.broadcast_to(w,(3,64)); b=jnp.tanh(u)
    plan,mass,cost,metrics=resampled_transport(key,jnp.tanh(mu),b,weights,.05,30,True)
    idx=multinomial_resample(key,weights,16)
    slots=jax.vmap(lambda x,i:x[i])(u,idx)
    P=sinkhorn(cost/(cost.mean(axis=(-2,-1),keepdims=True)+1e-8),jnp.ones((3,16))/16,.05,30)
    R=P/jnp.maximum(P.sum(axis=-1,keepdims=True),1e-20)
    collapsed=plan/jnp.maximum(plan.sum(axis=-1,keepdims=True),1e-20)
    ls=jnp.full_like(mu,-.7)
    direct=conditional_ot_nll(mu,ls,slots,R)
    aggregate=conditional_ot_nll(mu,ls,u,collapsed)
    np.testing.assert_allclose(direct,aggregate,rtol=2e-6)
    g1=jax.grad(lambda m,l:conditional_ot_nll(m,l,slots,R),argnums=(0,1))(mu,ls)
    g2=jax.grad(lambda m,l:conditional_ot_nll(m,l,u,collapsed),argnums=(0,1))(mu,ls)
    for x,y in zip(g1,g2):np.testing.assert_allclose(x,y,atol=2e-6)
    assert float(metrics['ot_slot_col_marginal_error'])<1e-6
    np.testing.assert_allclose(plan.sum(axis=-2),mass,atol=2e-6)
    assert all(np.isfinite(float(v)) for v in metrics.values())
    assert np.isclose(float(mass.sum(axis=-1).mean()),1)
    # NLL must not differentiate through either teacher values or assignment.
    gt=jax.grad(lambda t:conditional_ot_nll(mu,ls,t,collapsed))(u)
    gr=jax.grad(lambda r:conditional_ot_nll(mu,ls,u,r))(collapsed)
    assert np.max(np.abs(gt))==0 and np.max(np.abs(gr))==0
    # If all offspring duplicate one candidate, slot entropy must not be mistaken
    # for distinct-target diversity: 16 effective slots, one effective candidate.
    _,_,_,duplicate_metrics=resampled_transport(key,jnp.zeros((1,16,2)),jnp.zeros((1,64,2)),
        jax.nn.one_hot(jnp.array([7]),64),.05,30,True)
    np.testing.assert_allclose(duplicate_metrics['ot_slot_row_ess'],16,atol=1e-4)
    np.testing.assert_allclose(duplicate_metrics['ot_unique_row_ess'],1,atol=1e-5)
    np.testing.assert_allclose(duplicate_metrics['resample_unique_candidates'],1,atol=1e-5)


def test_v9_config_alias_and_environments():
    from run_optiq_dime import validate_config
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        default = compose(config_name="mujoco_v9")
        alias = compose(config_name="v9/final")
        assert OmegaConf.to_container(default, resolve=True) == OmegaConf.to_container(alias, resolve=True)
        for task in ["ant", "halfcheetah", "hopper", "walker2d", "humanoid"]:
            cfg = compose(config_name="mujoco_v9", overrides=[f"benchmark={task}"])
            validate_config(cfg)
            a = cfg.alg.actor
            assert a.sinkhorn_epsilon == .03 and a.sinkhorn_iterations == 50
            assert not a.normalize_ot_cost and a.temperature == .25
            assert a.num_policy_samples == 16 and a.proposals_per_policy_sample == 4
            assert a.proposal_sampling_mode == "exact" and not a.include_anchor
            assert a.distillation_loss == "conditional_ot_nll"
            assert cfg.alg.optimizer.ac_grad_norm is None
            assert cfg.alg.critic.backup_mode == "td"
            assert cfg.total_steps == 1000000 and cfg.dual_mu_eval
            assert cfg.experiment.resampling == "multinomial_iid_with_replacement"

def test_production_actor_update_changes_both_heads():
    from optiq_dime.algorithm import OptiQDIME
    from test_semi_implicit import actor_state, critic_state
    actor, critic = actor_state(), critic_state()
    out = OptiQDIME.update_actor(actor, critic, jnp.ones((4, 3)),
        jax.random.PRNGKey(19), jnp.array([-3600.]),
        16, 4, "exact", .05, .5, False, True, 1., False, 16., 257,
        .25, .03, 50, "mean", "argmax", True, False,
        "conditional_ot_nll", "conditional_mixture", 0., False, "mean")
    updated, loss, _, metrics = out[:4]
    assert np.isfinite(float(loss))
    assert float(metrics["resample_count"]) == 16
    assert float(metrics["ot_slot_col_marginal_error"]) < 1e-5
    for head in ["mu", "log_std"]:
        assert any(not np.array_equal(x, y) for x, y in zip(
            jax.tree_util.tree_leaves(actor.params[head]),
            jax.tree_util.tree_leaves(updated.params[head])))
