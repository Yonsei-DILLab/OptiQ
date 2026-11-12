"""Task interface for byte-identical official IDAC with the requested SIA actor."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

VENDOR = Path(__file__).resolve().parent/'vendor/idac'

class Agent:
    def __init__(self, config, folder, device='cuda'):
        manifest=json.loads((VENDOR/'SOURCE.json').read_text())
        for name,digest in manifest['files'].items():
            if hashlib.sha256((VENDOR/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('IDAC upstream source changed: '+name)
        sys.path.insert(0,str(VENDOR))
        cls=importlib.import_module('agents.idac').IDAC
        replay=importlib.import_module('utils.utils').ReplayBuffer
        self.device=torch.device(device)
        torch.manual_seed(config['seed'])
        self.config=dict(upstream=manifest['repository'],revision=manifest['revision'],
            pi_type='implicit',hidden_sizes=[256,256],noise_dim=5,num_quantiles=32,
            actor_noise_num=10,actor_noise_dim=5,actor_lr=3e-4,critic_lr=3e-4,
            discount=.99,tau=.005,alpha=.2,use_automatic_entropy_tuning=True,
            pi_bn=False,critic_layer_norm=True,batch_size=config['batch_size'],
            common_collector_warmup=config['warmup'],upstream_cli_warmup=10000)
        no_dgn=config['method']=='idac-nodgn'
        if no_dgn:
            class ScalarIDAC(cls):
                # Remove critic latent input and distributional regression.
                # Upstream actor, entropy, targets and optimizers are inherited.
                def quantile_regression_loss(self, input, target, tau, weight=1.):
                    return torch.nn.functional.mse_loss(input, target.detach())
                def get_tau(self, obs):
                    return torch.zeros((len(obs),1),device=self.device)
            cls=ScalarIDAC
            self.config.update(ablation='no DGN: deterministic scalar twin-Q with plain MSE',
                noise_dim=0,num_quantiles=1,distributional_critic=False,
                actor_unchanged=True,critic_layer_norm=True)
        self.agent=cls(state_dim=config['observation_dim'],action_dim=config['action_dim'],
            noise_dim=0 if no_dgn else 5,max_action=1.,device=self.device,hidden_sizes=[256,256],
            discount=.99,tau=.005,actor_lr=3e-4,critic_lr=3e-4,
            batch_size=config['batch_size'],pi_bn=False,num_quantiles=1 if no_dgn else 32,
            target_entropy=None,alpha=.2,use_automatic_entropy_tuning=True,pi_type='implicit')
        self.replay=replay(config['observation_dim'],config['action_dim'],self.device,
                           max_size=config['replay_capacity'])
        self.updates=0

    def act(self, obs, mode='train'):
        return np.stack([self.agent.sample_action(row) for row in obs]).astype(np.float32)

    def observe(self, obs, action, reward, next_obs, terminated):
        for fields in zip(obs,action,next_obs,reward,terminated):
            self.replay.add(*fields)

    def update(self, step):
        info=self.agent.train_from_batch(self.replay)
        if not all(np.isfinite(v).all() for v in info.values()):
            raise FloatingPointError('Nonfinite native IDAC loss')
        self.updates+=1

    @contextmanager
    def evaluation_rng(self,seed):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.device.type=='cuda' else []):
            torch.manual_seed(seed)
            if self.device.type=='cuda':
                torch.cuda.manual_seed_all(seed)
            yield

    def save(self,path):
        names=('actor','actor_target','gf1','gf2','gf1_target','gf2_target',
               'actor_optimizer','gf1_optimizer','gf2_optimizer','alpha_optimizer')
        state={name:getattr(self.agent,name).state_dict() for name in names}
        state.update(log_alpha=self.agent.log_alpha.detach(),updates=self.updates,config=self.config)
        torch.save(state,path)

    def save_replay(self,path):
        r=self.replay
        np.savez_compressed(path,state=r.state[:r.size],action=r.action[:r.size],
            next_state=r.next_state[:r.size],reward=r.reward[:r.size],not_done=r.not_done[:r.size],
            size=r.size,position=r.ptr)
