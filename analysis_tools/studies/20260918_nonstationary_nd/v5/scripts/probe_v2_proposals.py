"""Read-only importance-sampling diagnostics on saved screening checkpoints."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from flax import serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from optiq_dime import OptiQDIME
from optiq_dime.semi_implicit import actor_components,conditional_mixture_log_prob,PretanhTeacherKDE


def main():
    result=[]
    for directory in sorted((ROOT/'outputs/v2_screen').glob('*otnll*')):
        cfg=OmegaConf.load(directory/'config.json')
        model=OptiQDIME('MlpPolicy',gym.make('Humanoid-v4'),None,1,cfg)
        checkpoint=next(directory.glob('checkpoints/*/actor_state_25000.msgpack'))
        actor=serialization.from_bytes(model.policy.actor_state,checkpoint.read_bytes())
        critic=serialization.from_bytes(model.policy.qf_state,(checkpoint.parent/'critic_state_25000.msgpack').read_bytes())
        obs=jnp.asarray(np.load(checkpoint.parent/'landscape_probe_batch.npz')['observations'][:32])
        mu,ls=actor_components(actor,obs,jax.random.PRNGKey(91),16)
        centers=mu+jnp.exp(ls)*jax.random.normal(jax.random.PRNGKey(92),mu.shape)
        for count in (64,256):
            for proposal in ('conditional_mixture','kde_h0.4','kde_h0.6','kde_h0.8','kde_h1.0'):
                key=jax.random.PRNGKey(93)
                if proposal=='conditional_mixture':
                    ki,kn=jax.random.split(key)
                    indices=jax.random.randint(ki,(len(obs),count),0,16)
                    selected_mu=jnp.take_along_axis(mu,indices[:,:,None],axis=1)
                    selected_ls=jnp.take_along_axis(ls,indices[:,:,None],axis=1)
                    v=selected_mu+jnp.exp(selected_ls)*jax.random.normal(kn,selected_mu.shape)
                    logq=conditional_mixture_log_prob(v,mu,ls)
                else:
                    kde=PretanhTeacherKDE(centers,float(proposal.split('h')[1]))
                    _,v,_=kde.sample(key,count//16,'exact')
                    logq=kde.log_prob(v)
                actions=jnp.tanh(v)
                repeated=jnp.broadcast_to(obs[:,None],(len(obs),count,obs.shape[-1])).reshape(len(obs)*count,-1)
                q=critic.apply_fn({'params':critic.params,'batch_stats':critic.batch_stats},repeated,
                    actions.reshape(len(obs)*count,-1),rngs={'dropout':key},train=False)[...,0]
                q=q.mean(axis=0).reshape(len(obs),count)
                weights=jax.nn.softmax(q/cfg.alg.actor.temperature-logq,axis=-1)
                dw=jax.nn.softmax(-logq,axis=-1)
                ess=1/(weights**2).sum(-1)
                row={'run':directory.name,'temperature':float(cfg.alg.actor.temperature),'proposal':proposal,
                    'count':count,'ess_mean':float(ess.mean()),'density_only_ess_mean':float((1/(dw**2).sum(-1)).mean()),
                    'ess_p10':float(jnp.quantile(ess,.1)),'std_mu_across_z':float(mu.std(axis=1).mean()),
                    'sigma_mean':float(jnp.exp(ls).mean()),'mu_abs_mean':float(jnp.abs(mu).mean())}
                result.append(row);print(json.dumps(row),flush=True)
        model.get_env().close()
    (ROOT/'outputs/v2_improvement/proposal_probe.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
