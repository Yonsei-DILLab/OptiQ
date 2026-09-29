"""CPU-only forward diagnostics on preserved checkpoints. Never updates weights."""
import os
os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1', USE_FLAX='0', USE_TORCH='1')
import sys
from pathlib import Path
import json
import hashlib
import gc
import numpy as np
import torch
torch.set_num_threads(1)
import jax
import jax.numpy as jnp
import flax.serialization
import flax.linen as nn

source = Path(sys.argv[1])
run = Path(sys.argv[2])
config = json.loads((run / 'config.json').read_text())
checkpoint = torch.load(run / 'checkpoint-final.pt', map_location='cpu', weights_only=False)
assert checkpoint['config']['source_commit'] == config['source_commit']
state = checkpoint['learner']
indices = np.random.default_rng(923).choice(len(checkpoint['replay']['buf_obs']), 128, replace=False)
obs = jnp.asarray(checkpoint['replay']['buf_obs'][indices].numpy())
rewards = checkpoint['replay']['buf_reward'].numpy()
result = dict(run=run.name, source=config['source_commit'], checkpoint_step=checkpoint['step'],
              checkpoint_updates=checkpoint['updates'], diagnostic_seed=923, states=128,
              candidates_per_state=64, compute='CPU forward only; no optimizer or environment steps',
              stored_nonzero_rewards=int(np.count_nonzero(rewards)))
del checkpoint, rewards
gc.collect()
B, N, A = 128, 64, 8
repeated = jnp.repeat(obs, N, axis=0)
key = jax.random.PRNGKey(923)

def spread(actions):
    a = np.asarray(actions).reshape(B, N, A)
    return dict(rms_coordinate_std=float(np.sqrt(np.var(a, axis=1).mean())),
                mean_spread_l2=float(np.linalg.norm(a.std(axis=1),axis=-1).mean()),
                abs_mean=float(np.abs(a).mean()),
                saturation_fraction=float((np.abs(a)>.99).mean()))

def q_stats(q, temp):
    q = np.asarray(q)
    return dict(mean=float(q.mean()), candidate_std_mean=float(q.std(-1).mean()),
                candidate_range_mean=float(np.ptp(q,axis=-1).mean()),
                logit_std_mean=float((q/temp).std(-1).mean()))

if config['method'] == 'mfpo':
    import gym
    sys.path.insert(0, str(source / 'gmm40-baseline/MFPO'))
    from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
    from jaxrl5.networks.mean_flow import action_sampler_with_logp
    template = MeanFlowLearner.create(0,
        gym.spaces.Box(-1,1,(29,),dtype=np.float32),
        gym.spaces.Box(-1,1,(8,),dtype=np.float32), **config['native'])
    agent = flax.serialization.from_bytes(template, state['agent'])
    temperature = float(agent.temp.apply_fn({'params':agent.temp.params}))
    action, logp = action_sampler_with_logp(agent.actor.apply_fn, agent.actor.params,
        agent.logp_mvel.apply_fn, agent.logp_mvel.params, agent.T,
        jax.random.normal(key,(B*N,A)), repeated, agent.clip_sampler)
    probs = []
    for c in (agent.critic_1,agent.critic_2):
        logits = c.apply_fn({'params':c.params}, repeated, action)
        probs.append(jax.nn.softmax(logits,-1))
    qs = [np.asarray((p*agent.z_atoms).sum(-1)).reshape(B,N) for p in probs]
    q = np.minimum(*qs)
    result.update(temperature=temperature, target_entropy=agent.target_entropy,
        logp_mean=float(logp.mean()), policy_spread=spread(action),
        q=q_stats(q,temperature), critic_support=[agent.v_min,agent.v_max],
        atom_spacing=agent.delta_z,
        mass_zero_atom=float(np.mean([np.asarray(p)[:,50].mean() for p in probs])),
        mass_central_three_atoms=float(np.mean([np.asarray(p)[:,49:52].sum(-1).mean() for p in probs])),
        q_only_ess=float(np.mean(1/np.square(np.asarray(jax.nn.softmax(jnp.asarray(q)/temperature,-1))).sum(-1))))
else:
    sys.path.insert(0,str(source / 'analysis_tools/studies/20260918_nonstationary_nd/v5'))
    sys.path.insert(0,str(source / 'analysis_tools/experiments/20260920_truncated_mll'))
    from optiq_dime.policy import SemiImplicitActor
    from optiq_dime.semi_implicit import ConditionalGaussianProposal
    from optiq_dime.box_gaussian import sample_box
    from models.critic import VectorCritic
    from models.utils import activation_fn
    cfg=config['native']['alg']; a=cfg['actor']; c=cfg['critic']; o=cfg['optimizer']
    raw=flax.serialization.msgpack_restore(state['policy'])
    actor=SemiImplicitActor(A,tuple(a['hidden_dims']),a['log_std_min'],a['log_std_max'],
        a['initial_log_std'],a['mean_output_init_scale'],a['log_std_output_init_scale'],a.get('mean_latent_skip_scale',0.))
    (mu,logstd), intermediate=actor.apply({'params':raw['actor']['params']},repeated,
        jax.random.normal(key,(B*N,A)), capture_intermediates=lambda m,n:m.name=='log_std',
        mutable=['intermediates'])
    raw_logstd=np.asarray(intermediate['intermediates']['log_std']['__call__'][0])
    mu=mu.reshape(B,N,A); logstd=logstd.reshape(B,N,A)
    proposal=ConditionalGaussianProposal(mu,logstd,a['proposal_std'])
    actions, u, components=proposal.sample(jax.random.PRNGKey(924),1,'exact')
    actions=actions.reshape(B,N,A); u=u.reshape(B,N,A)
    density=proposal.log_prob(u)
    critic=VectorCritic(net_arch=c['hs'],activation_fn=activation_fn[c['activation']],
        batch_norm_momentum=o['bn_momentum'],bn_warmup=o['bn_warmup'],use_batch_norm=o['bn'],
        batch_norm_mode=o['bn_mode'],use_layer_norm=c['use_layer_norm'],dropout_rate=c['dropout_rate'],
        n_critics=c['n_critics'],n_atoms=c['n_atoms'])
    qs=np.asarray(critic.apply({'params':raw['critic']['params'],'batch_stats':raw['critic']['batch_stats']},
        repeated,actions.reshape(B*N,A),train=False)).reshape(2,B,N)
    q=qs.mean(0); temperature=a['temperature']
    d=np.asarray(density)
    density_weights=np.asarray(jax.nn.softmax(-density,axis=-1))
    controls=[]
    for t in [0.25,0.05,0.01,0.005,0.001]:
        w=np.asarray(jax.nn.softmax(jnp.asarray(q)/t-density,axis=-1))
        q_w=np.asarray(jax.nn.softmax(jnp.asarray(q)/t,axis=-1))
        controls.append(dict(temperature=t,
            weight_tv_from_density_only=float(np.abs(w-density_weights).sum(-1).mean()/2),
            q_only_ess=float((1/np.square(q_w).sum(-1)).mean()),
            full_ess=float((1/np.square(w).sum(-1)).mean()),
            full_weighted_q_gain=float(((w*q).sum(-1)-q.mean(-1)).mean())))
    result.update(temperature=temperature,q=q_stats(q,temperature),
        policy_spread=spread(sample_box(jax.random.PRNGKey(925),mu,logstd)),
        teacher_candidate_spread=spread(actions), mu_spread=spread(mu),
        sigma_mean=float(np.exp(np.asarray(logstd,dtype=np.float64)).mean()),sigma_cap_fraction=float((logstd>=-1.-1e-6).mean()),
        raw_logstd_min=float(raw_logstd.min()),raw_logstd_mean=float(raw_logstd.mean()),raw_logstd_max=float(raw_logstd.max()),
        raw_logstd_strictly_above_cap_fraction=float((raw_logstd>-1).mean()),
        density_logit_std=float(d.std(-1).mean()),
        density_to_q_logit_std=float(d.std(-1).mean()/(q/temperature).std(-1).mean()),
        same_checkpoint_temperature_reweighting=controls)

centered=qs[0]-qs[0].mean(-1,keepdims=True)
other=qs[1]-qs[1].mean(-1,keepdims=True)
result['twin_action_q_correlation_mean']=float(np.mean((centered*other).mean(-1)/(centered.std(-1)*other.std(-1)+1e-12)))
assert all(np.isfinite(np.asarray(q)).ravel())
print('AUDIT_JSON='+json.dumps(result,allow_nan=False))
