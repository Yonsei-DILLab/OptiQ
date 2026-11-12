"""SB3 SAC with LayerNorm after each hidden critic Linear and before ReLU."""
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import SACPolicy
from torch import nn


class LayerNormCritic(ContinuousCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for index, network in enumerate(self.q_networks):
            layers = []
            for layer in list(network.children())[:-1]:
                layers.append(layer)
                if isinstance(layer, nn.Linear):
                    layers.append(nn.LayerNorm(layer.out_features, eps=1e-6))
            layers.append(network[-1])
            q = nn.Sequential(*layers)
            self.add_module(f"qf{index}", q)
            self.q_networks[index] = q


class LayerNormSACPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormCritic(**kwargs).to(self.device)
