"""Common runner interface around private historical native baseline adapters."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


def verify_sources():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'SOURCE.json').read_text())
    if hashlib.sha256((root/'legacy_agents.py').read_bytes()).hexdigest() != manifest['legacy_agents_adapted_sha256']:
        raise RuntimeError('Frozen baseline adapter changed')
    for name, vendor in manifest['vendors'].items():
        for file, digest in vendor['files'].items():
            if hashlib.sha256((root/'vendor'/name/file).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Vendor source changed: ' + name + '/' + file)
    if hashlib.sha256((root/'sql_jax.py').read_bytes()).hexdigest() != manifest['sql_jax_sha256']:
        raise RuntimeError('Frozen SQL implementation changed')
    return manifest


class Agent:
    evaluation_mode = 'policy'

    def __init__(self, config, output, device='cuda'):
        import jax
        import torch
        from .legacy_agents import SAC, TD3, SQL, MFPO, MEOW
        self.method = config['method']
        self.output = output
        self.count = 0
        self.sources = verify_sources()
        if device == 'cuda' and self.method in ('sql', 'mfpo') and jax.default_backend() != 'gpu':
            raise RuntimeError('JAX GPU required')
        if device == 'cuda' and self.method in ('sac', 'td3', 'meow') and not torch.cuda.is_available():
            raise RuntimeError('PyTorch GPU required')
        settings = config['native_settings']
        kwargs = {}
        if self.method in ('sac', 'td3', 'meow'):
            kwargs['device'] = device
        if self.method == 'sql':
            kwargs.update(temperature=settings['sql_temperature'], particles=settings['sql_particles'])
        elif self.method == 'meow':
            kwargs['alpha'] = settings['meow_alpha']
        elif self.method == 'mfpo':
            kwargs['target_entropy_coeff'] = settings['mfpo_target_entropy_coeff']
        cls = dict(sac=SAC, td3=TD3, sql=SQL, mfpo=MFPO, meow=MEOW)[self.method]
        self.native = cls(config['seed'], output, config['steps'], 4,
                          batch_size=config['batch_size'], **kwargs)
        # Allocate the requested replay shape, including bounded smoke validation.
        from ..envs.replay import Replay
        old = self.native.replay
        self.native.replay = Replay(capacity=config['replay_capacity'], seed=config['seed'], device=old.device)
        if self.method in ('sac', 'td3'):
            self.native.model.replay_buffer = self.native.replay
        native_config = self.native.config
        self.config = asdict(native_config) if is_dataclass(native_config) else dict(native_config)
        self.config.update(native_policy_evaluation='direct native sampling; TD3 deterministic',
                           frozen_adapter_revision=self.sources['revision'])

    @property
    def updates(self):
        return self.count

    def observe(self, obs, actions, rewards, next_obs, done):
        self.native.replay.add(obs, actions, rewards, next_obs, done.astype(np.float32))

    def act(self, observations, mode='train'):
        if mode not in ('train', 'policy'):
            raise ValueError('Baseline uses native policy evaluation')
        return self.native.act(observations, mode=mode)

    def evaluation_rng(self, seed):
        return self.native.evaluation_rng(seed)

    def update(self, step):
        self.native.update(step)
        self.count += 1
        if self.native.updates != self.count:
            raise RuntimeError('Native optimizer update count mismatch')

    def assert_finite(self):
        import jax
        import torch
        if self.method in ('sac', 'td3'):
            arrays = list(self.native.model.policy.parameters())
            if getattr(self.native.model, 'log_ent_coef', None) is not None:
                arrays.append(self.native.model.log_ent_coef)
        elif self.method == 'meow':
            arrays = list(self.native.policy.parameters()) + list(self.native.target.parameters())
        else:
            state = self.native.learner.state if self.method == 'sql' else self.native.agent
            arrays = jax.tree_util.tree_leaves(state)
        for value in arrays:
            finite = bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else np.isfinite(np.asarray(value)).all()
            if not finite:
                raise FloatingPointError('Nonfinite baseline model state')

    def save(self, path):
        saved = self.native.save(path.parent, int(path.stem), full=False)
        if saved != path:
            saved.rename(path)

    def save_replay(self, path):
        replay = self.native.replay
        np.savez_compressed(path, **{k: v[:replay.size] for k, v in replay.data.items()},
                            size=replay.size, position=replay.position)

    def flush_logs(self, step):
        if self.method in ('sac', 'td3'):
            logger = self.native.model.logger
            logger.record('env_steps', step)
            logger.dump(step)

    def close(self):
        if self.method in ('sac', 'td3'):
            self.native.model.get_env().close()
            self.native.model.logger.close()
