"""Two MFPO scalar overrides and separately labeled official evaluation."""
import numpy as np
from ..baselines.adapter import Agent as NativeAgent


class Agent(NativeAgent):
    evaluation_mode = 'official'

    def __init__(self, config, output, device='cuda'):
        import jax
        import flax.serialization
        import hashlib
        super().__init__(config, output, device=device)
        # NativeAgent imports the verified, pinned vendor before this module.
        from jaxrl5.networks import Temperature
        initial = float(config['initial_temperature'])
        def unchanged_payload(agent):
            return flax.serialization.to_bytes(dict(actor=agent.actor,logp=agent.logp_mvel,
                critic1=agent.critic_1,critic2=agent.critic_2,
                target1=agent.target_critic_1,target2=agent.target_critic_2,
                rng=agent.rng,temp_optimizer=agent.temp.opt_state))
        before=hashlib.sha256(unchanged_payload(self.native.agent)).hexdigest()
        definition = Temperature(initial)
        params = definition.init(jax.random.PRNGKey(config['seed'] + 771901))['params']
        old = self.native.agent.temp
        if jax.tree_util.tree_structure(old.params) != jax.tree_util.tree_structure(params):
            raise RuntimeError('Initial temperature parameter structure changed')
        self.native.agent = self.native.agent.replace(temp=old.replace(
            params=params, apply_fn=definition.apply))
        after=hashlib.sha256(unchanged_payload(self.native.agent)).hexdigest()
        if before!=after:
            raise RuntimeError('Temperature override changed other learner parameters or RNG')
        self.initialization_audit=dict(unchanged_actor_critic_logp_rng_and_temp_optimizer=True,
            unchanged_state_sha256=after,initial_temperature=initial,
            target_entropy=float(self.native.agent.target_entropy))
        # This is a scalar initialization change. Preserve optimizer state,
        # training RNG and every actor/critic/log-density parameter.
        self.native.config['temp'] = initial
        self.config.update(self.native.config)
        self.config.update(evaluation_primary='official 10-candidate target-Q selection',
            evaluation_secondary='single direct policy sample',
            training_policy='unchanged native direct sampling',
            initial_temperature=initial)
        if self.native.agent.eval_candidate_num != 10:
            raise RuntimeError('Official candidate count must stay 10')

        @jax.jit
        def select(agent, observations):
            keys = jax.random.split(agent.rng, len(observations) + 1)
            actions = jax.vmap(lambda state, key:
                agent.replace(rng=key).eval_actions_select(state, cand_num=10)[0])(
                    observations, keys[1:])
            return actions, agent.replace(rng=keys[0])
        self.select_fn = select
        self.last_info = {}

    def act(self, observations, mode='train'):
        observations = np.asarray(observations, np.float32)
        if mode == 'official':
            actions, self.native.agent = self.select_fn(self.native.agent, observations)
            return np.clip(np.asarray(actions), -1., 1.).astype(np.float32)
        if mode in ('train', 'single_sample'):
            return self.native.act(observations, mode='policy')
        raise ValueError('Unknown MFPO evaluation mode')

    def update(self, step):
        self.last_info = self.native.update(step)
        self.count += 1
        if self.native.updates != self.count:
            raise RuntimeError('MFPO optimizer update count mismatch')
