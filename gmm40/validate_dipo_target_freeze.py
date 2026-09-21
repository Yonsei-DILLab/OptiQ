"""Temporary paired updates to test freezing inference-only target parameters."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluation import atomic_json
from .navigation import GMM40Navigation
from .navigation_agents import DIPOOnline
from .target import RESULTS


def identical(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(identical(a,b) for a,b in zip(left,right))
    return left == right


def main():
    checkpoint = RESULTS/'validation_navigation_dipo_native_100/checkpoints/update_0000100.bin'
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    runs = {}
    for freeze in (False, True):
        env = GMM40Navigation()
        wrapper = DIPOOnline(env, 0, checkpoint.parent.parent)
        agent = wrapper.agent
        for name in ('actor','actor_target','critic','critic_target'):
            getattr(agent,name).load_state_dict(state[name])
        for name in ('actor_optimizer','critic_optimizer'):
            getattr(agent,name).load_state_dict(copy.deepcopy(state[name]))
        wrapper.memory.__dict__.update(copy.deepcopy(state['memory']))
        wrapper.dmemory.__dict__.update(copy.deepcopy(state['diffusion_memory']))
        agent.step = state['updates']
        if freeze:
            agent.actor_target.requires_grad_(False)
            agent.critic_target.requires_grad_(False)
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        for _ in range(2):
            agent.train(1, 256)
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start = time.perf_counter()
            agent.train(1, 256)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        actions = wrapper.act(state['memory']['states'][:8])
        count = wrapper.dmemory.capacity if wrapper.dmemory.full else wrapper.dmemory.idx
        runs[freeze] = dict(
            models={name:copy.deepcopy(getattr(agent,name).state_dict())
                    for name in ('actor','actor_target','critic','critic_target')},
            optimizers={name:copy.deepcopy(getattr(agent,name).state_dict())
                        for name in ('actor_optimizer','critic_optimizer')},
            best_actions=wrapper.dmemory.best_actions[:count].copy(),
            collection_actions=actions.copy(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            step=agent.step, times=times)
        env.close()
    checks = {key:identical(runs[False][key],runs[True][key])
              for key in ('models','optimizers','best_actions','collection_actions','numpy_rng','torch_rng','cuda_rng','step')}
    original = float(np.median(runs[False]['times']))
    frozen = float(np.median(runs[True]['times']))
    result = dict(checkpoint=str(checkpoint), input_checkpoint_unchanged=before==hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  temporary_validation_updates_per_variant=22, main_training_modified=False,
                  bitwise_identical=checks, all_checks_passed=all(checks.values()),
                  median_native_update_seconds=original, median_frozen_target_update_seconds=frozen,
                  update_speedup=original/frozen,
                  scope='One saved validation learner and 22 paired native updates plus sampled collection actions. Timing excludes live action collection, evaluation, and file writing; no current run is modified.')
    atomic_json(RESULTS/'diagnostics/dipo_target_freeze_validation.json',result)
    print(json.dumps(result,indent=2))
    if not result['all_checks_passed'] or not result['input_checkpoint_unchanged']:
        raise RuntimeError('Paired target-freeze validation failed')


if __name__ == '__main__':
    main()
