"""CPU constructor audits for every entropy grid value; no learner training."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+',required=True);args=parser.parse_args()
    import torch
    import numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from rebuttal_tools.fetch_entropy_campaign import GRIDS
    from rebuttal_tools.fetch_entropy_worker import install
    from fetch_rebuttal.baselines.native import SpaceOnlyEnv
    reports=[]
    for method in args.methods:
        for value in GRIDS[method][1]:
            with tempfile.TemporaryDirectory() as directory:
                spec=dict(method=method,parameter=GRIDS[method][0],value=value,phase='cpu-check',
                    task='slide',seed=100,output=directory,temperature=value if method=='ibolt' else .25,
                    mfpo_target_entropy_coeff=value if method=='mfpo' else 0.,mfpo_initial_temperature=1.)
                module=install(spec)
                try:
                    if method=='sac':
                        from stable_baselines3 import SAC
                        model=SAC('MlpPolicy',SpaceOnlyEnv(28,4),seed=100,device='cpu',
                            policy_kwargs=dict(net_arch=[256,256]),learning_rate=3e-4,
                            buffer_size=256,ent_coef='auto')
                        assert model.target_entropy==value and model._n_updates==0
                        assert model.ent_coef_optimizer is not None
                        loss=-(model.log_ent_coef*(torch.full((8,),-1.)+model.target_entropy).detach()).mean()
                        model.ent_coef_optimizer.zero_grad();loss.backward()
                        alpha_gradient=float(model.log_ent_coef.grad)
                        assert abs(alpha_gradient-(1-value))<1e-6
                        detail=dict(target_entropy=model.target_entropy,automatic_alpha=True,
                            initial_alpha=float(model.log_ent_coef.detach().exp()),alpha_gradient=alpha_gradient)
                    elif method=='meow':
                        from fetch_rebuttal.baselines.native import MEOW
                        model=MEOW(100,Path(directory),5000192,28,batch_size=256,device='cpu',action_dim=4)
                        assert model.policy.alpha==model.target.alpha==value and model.updates==0
                        obs=torch.zeros((8,28));act=torch.zeros((8,4))
                        model.policy.eval()
                        with torch.no_grad():
                            q,v=model.policy.get_qv(obs,act);model.policy.alpha=1.
                            q0,v0=model.policy.get_qv(obs,act);model.policy.alpha=value
                        torch.testing.assert_close(q,q0*value);torch.testing.assert_close(v,v0*value)
                        detail=dict(alpha=value,target_alpha=model.target.alpha,qv_scaling_verified=True)
                    elif method=='mfpo':
                        import sys,jax
                        import fetch_rebuttal.baselines.native as native
                        vendor=Path(native.__file__).parent/'vendor/mfpo'
                        loader=importlib.util.spec_from_file_location('mfpo_entropy_config',vendor/'configs/mfpo_config.py')
                        cfgmodule=importlib.util.module_from_spec(loader);loader.loader.exec_module(cfgmodule)
                        cfg=cfgmodule.get_config().to_dict();cfg.pop('model_cls')
                        cfg.update(actor_hidden_dims=(256,256),critic_hidden_dims=(256,256),actor_layer_norm=False,
                            critic_layer_norm=False,target_entropy_coeff=value,temp=1.)
                        sys.path.insert(0,str(vendor))
                        from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
                        env=SpaceOnlyEnv(28,4);model=MeanFlowLearner.create(100,env.observation_space,env.action_space,**cfg)
                        assert float(model.target_entropy)==4*value and int(model.actor.step)==0
                        assert np.isclose(float(model.temp.apply_fn({'params':model.temp.params})),1.)
                        from flax.traverse_util import flatten_dict
                        for state in [model.actor,model.critic_1,model.critic_2,model.logp_mvel]:
                            assert not any('LayerNorm' in str(key) for key in flatten_dict(state.params))
                        assert model.actor.params['MLP_0']['Dense_1']['kernel'].shape==(256,256)
                        assert len([k for k in model.actor.params['MLP_0'] if k.startswith('Dense_')])==3
                        detail=dict(target_entropy=float(model.target_entropy),initial_temperature=1.,
                            actor_hidden_dims=[256,256],critic_hidden_dims=[256,256],layer_norm=False)
                    else:
                        from train import compose_config
                        cfg=compose_config(['benchmark=ant','wandb.activate=false'])
                        cfg.temperature=value
                        assert float(cfg.alg.actor.temperature)==value
                        assert list(cfg.alg.actor.hidden_dims)==list(cfg.alg.critic.hs)==[256,256]
                        detail=dict(temperature=float(cfg.alg.actor.temperature),actor_hidden_dims=[256,256])
                    reports.append(dict(method=method,value=value,passed=True,learner_updates=0,**detail))
                finally:module.restore_entropy_override()
    print(json.dumps(dict(passed=True,cases=reports)))


if __name__=='__main__':main()
