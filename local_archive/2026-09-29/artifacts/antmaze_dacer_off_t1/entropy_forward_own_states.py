"""CPU-only entropy/noise calibration of preserved policies; no RL updates."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',
    OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',USE_FLAX='0',USE_TORCH='1')
cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,(cpus[2::4] or cpus)[:4])
import sys,json,hashlib,gc,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
import jax
import jax.numpy as jnp
import flax.serialization
from sklearn.mixture import GaussianMixture

BASE=Path('/home/heechan/optiq-experiments')
SOURCE=Path('/home/heechan/OptiQ-ops/sources/484f92e7d6d34c964d85b4493ff17c5a9ebcf32e')
MFPO_SOURCE=Path('/home/heechan/OptiQ-ops/sources/05fc113aeb093f9f83208ca540dd2aadf8d934bb')
sys.path.insert(0,str(SOURCE/'antmaze'))
sys.path.insert(0,str(SOURCE/'analysis_tools/studies/20260918_nonstationary_nd/v5'))
sys.path.insert(0,str(SOURCE/'analysis_tools/experiments/20260920_truncated_mll'))
from optiq_dime.policy import SemiImplicitActor
from optiq_dime.box_gaussian import sample_box

B,N,A=24,200,8
TASKS=('v3',)
rng=np.random.default_rng(240924)
output=dict(compute='CPU inference only; no optimizer/environment steps',states=B,
    fit_samples_per_state=N,heldout_samples_per_state=N,seed=240924,
    model_interface_source=str(SOURCE),inputs={},results={})


def checkpoint(run):
    cfg=json.loads((run/'config.json').read_text());p=run/'checkpoint-final.pt'
    proof=json.loads((run/'checkpoint-verification.json').read_text())
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest==proof['sha256'] and proof['readback_verified']
    x=torch.load(p,map_location='cpu',weights_only=False)
    assert x['config']['source_commit']==cfg['source_commit'] and x['step']==cfg['steps']
    output['inputs'][str(p)]=dict(sha256=digest,source=cfg['source_commit'],step=x['step'])
    return cfg,x


def metrics(actions):
    actions=np.asarray(actions,dtype=np.float64).reshape(B,2*N,A)
    assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1.00001
    joint=[];marginal=[];cross=[];gap=[];outside=[];converged=[]
    for row in actions:
        g=GaussianMixture(3,covariance_type='full',random_state=42).fit(row[:N])
        sign,ld=np.linalg.slogdet(g.covariances_);assert np.all(sign>0)
        w=g.weights_;h=-np.dot(w,np.log(w))+np.dot(w,.5*(A*(1+np.log(2*np.pi))+ld))
        sampled,_=g.sample(8000);post=g.predict_proba(sampled)
        conditional=-np.mean(np.sum(post*np.log(np.maximum(post,1e-300)),axis=1))
        joint.append(h);gap.append(conditional);marginal.append(h-conditional)
        cross.append(-g.score_samples(row[N:]).mean())
        outside.append(np.any(np.abs(sampled)>1,axis=1).mean());converged.append(g.converged_)
    return dict(action_rms_coordinate_std=float(np.sqrt(actions.var(axis=1).mean())),
        mean_abs_action=float(np.abs(actions).mean()),saturation_above_099=float((np.abs(actions)>.99).mean()),
        boundary_atom_fraction=float((np.abs(actions)>=1).mean()),
        gmm_joint_proxy_per_dim=float(np.mean(joint)/A),
        fitted_gmm_marginal_entropy_per_dim_mc=float(np.mean(marginal)/A),
        gmm_joint_minus_marginal_per_dim_mc=float(np.mean(gap)/A),
        heldout_policy_cross_entropy_under_gmm_per_dim=float(np.mean(cross)/A),
        fitted_gmm_outside_box_fraction=float(np.mean(outside)),
        gmm_converged_fraction=float(np.mean(converged)),
        per_state_joint_proxy_per_dim=(np.asarray(joint)/A).tolist())


for task in TASKS:
    run=BASE/'antmaze-optiq-dense-off-T1-s0-20260924/runs'/f'{task}-optiq-s0'
    cfg,saved=checkpoint(run)
    index=rng.choice(len(saved['replay']['buf_obs']),B,replace=False)
    obs=saved['replay']['buf_obs'][index].numpy().copy();state=saved['learner']
    obs_hash=hashlib.sha256(obs.tobytes()).hexdigest()
    del saved;gc.collect()
    a=cfg['native']['alg']['actor'];raw=flax.serialization.msgpack_restore(state['policy'])
    actor=SemiImplicitActor(A,tuple(a['hidden_dims']),a['log_std_min'],a['log_std_max'],
        a['initial_log_std'],a['mean_output_init_scale'],a['log_std_output_init_scale'],a.get('mean_latent_skip_scale',0.))
    repeated=jnp.repeat(jnp.asarray(obs),2*N,axis=0)
    mu,logstd=actor.apply({'params':raw['actor']['params']},repeated,
        jax.random.normal(jax.random.PRNGKey(240924),(B*2*N,A)))
    actions=np.asarray(sample_box(jax.random.PRNGKey(240925),mu,logstd)).reshape(B,2*N,A)
    noise=rng.normal(size=actions.shape)
    row=dict(method='optiq',task=task,source=cfg['source_commit'],state_pool_sha256=obs_hash,
        state_pool='24 uniform samples from same-task completed OptiQ T1 final replay',
        sigma_parameter_mean=float(np.exp(np.asarray(logstd)).mean()),
        sigma_at_upper_cap_fraction=float((np.asarray(logstd)>=-1.000001).mean()),
        mu_rms_coordinate_std=float(np.sqrt(np.asarray(mu).reshape(B,2*N,A).var(axis=1).mean())),
        policy=metrics(actions),noise_grid={})
    for std in (.02,.05,.1,.2,.35):
        row['noise_grid'][str(std)]=metrics(np.clip(actions+std*noise,-1,1))
    output['results'][task+'-optiq']=row
    print('PROGRESS '+task+'-optiq',file=sys.stderr,flush=True)
    if task!='v3':continue
    # Compare baselines on the identical OptiQ replay states, not different state distributions.
    torch_obs=torch.from_numpy(np.repeat(obs,2*N,axis=0))
    for method in ('sac','dipo','mfpo'):
        campaign='antmaze-dense-anneal-baselines-s0-20260924' if method=='mfpo' else 'antmaze-dense-off-16-current-s0-20260924'
        other=BASE/campaign/'runs'/f'{task}-{method}-s0'
        bc,loaded=checkpoint(other);bs=loaded['learner']
        bi=rng.choice(len(loaded['replay']['buf_obs']),B,replace=False)
        obs=loaded['replay']['buf_obs'][bi].numpy().copy()
        obs_hash=hashlib.sha256(obs.tobytes()).hexdigest()
        repeated=jnp.repeat(jnp.asarray(obs),2*N,axis=0)
        torch_obs=torch.from_numpy(np.repeat(obs,2*N,axis=0))
        del loaded;gc.collect()
        extra={};torch.manual_seed(240924)
        if method=='sac':
            from ddiffpg.models.mlp import TanhDiagGaussianMLPPolicy
            model=TanhDiagGaussianMLPPolicy((29,),8);model.load_state_dict(bs['actor']);model.eval()
            assert not bc['native']['algo']['obs_norm']
            with torch.no_grad():
                loc,ls=model.net(torch_obs).chunk(2,-1);std=ls.clamp(-5,5).exp()
                u=loc+std*torch.randn_like(loc);act=u.tanh()
                logjac=2*(np.log(2)-u-torch.nn.functional.softplus(-2*u))
                lp=(torch.distributions.Normal(loc,std).log_prob(u)-logjac).sum(-1)
            arr=act.numpy().reshape(B,2*N,A)
            extra.update(exact_squashed_gaussian_entropy_per_dim_mc=float(-lp.mean()/8),
                pre_tanh_sigma_mean=float(std.mean()),target_entropy_per_dim=-1.)
        elif method=='dipo':
            from ddiffpg.models.diffusion_mlp import DiffusionPolicy
            model=DiffusionPolicy((29,),8,bc['native']['diffusion']['diffusion_iter'],device='cpu')
            model.load_state_dict(bs['actor']);model.eval()
            assert not bc['native']['algo']['obs_norm']
            with torch.no_grad():arr=model(torch_obs).numpy().reshape(B,2*N,A)
            nc=bc['native']['algo']['noise'];assert nc['type']=='mixed'
            stds=np.linspace(nc['std_min'],nc['std_max'],2*N)[None,:,None]
            behavior=np.clip(arr+stds*rng.normal(size=arr.shape),-1,1)
            extra.update(training_mixed_noise_range=[nc['std_min'],nc['std_max']],
                training_mixed_noise_rms=float(np.sqrt(np.mean(np.linspace(nc['std_min'],nc['std_max'],256)**2))),
                population_behavior_with_mixed_noise=metrics(behavior))
        else:
            import gym
            sys.path.insert(0,str(MFPO_SOURCE/'gmm40-baseline/MFPO'))
            from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
            from jaxrl5.networks.mean_flow import action_sampler_with_logp
            template=MeanFlowLearner.create(0,gym.spaces.Box(-1,1,(29,),dtype=np.float32),
                gym.spaces.Box(-1,1,(8,),dtype=np.float32),**bc['native'])
            agent=flax.serialization.from_bytes(template,bs['agent'])
            act,lp=action_sampler_with_logp(agent.actor.apply_fn,agent.actor.params,
                agent.logp_mvel.apply_fn,agent.logp_mvel.params,agent.T,
                jax.random.normal(jax.random.PRNGKey(240924),(B*2*N,A)),repeated,agent.clip_sampler)
            arr=np.asarray(act).reshape(B,2*N,A)
            extra.update(learned_logp_entropy_per_dim=float(-lp.mean()/8),
                target_entropy_per_dim=float(agent.target_entropy)/8)
        output['results'][task+'-'+method]=dict(method=method,task=task,source=bc['source_commit'],
            state_pool_sha256=obs_hash,state_pool='24 random states from each baseline own final replay',policy=metrics(arr),**extra)
        print('PROGRESS '+task+'-'+method,file=sys.stderr,flush=True)
    del state,raw;gc.collect()

output['limitations']=['24 states per maze; one training seed; no new rollout or learning.',
    'Baseline diagnostics use own replay states; different state distributions prevent a causal cross-method comparison.',
    'GMM joint entropy is an upper bound on fitted-mixture entropy; marginal estimate is not true-policy entropy.',
    'Held-out GMM cross entropy includes density-fit error; clipped behavior has boundary atoms.',
    'Noise grid is a frozen-policy diagnostic, not an online training outcome.',
    'MFPO learned log-density and SAC analytic density are separate estimators, not DACER proxy values.']
print('AUDIT_JSON='+json.dumps(output,allow_nan=False))
