"""Actual learner parity and timed GPU comparison of metric logging modes."""
import argparse
import copy
import csv
import json
from pathlib import Path
import time

from .guards import assert_work_branch, assert_output_boundary, code_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2048)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--collection-execution', choices=('single', 'scan'), default='single')
    args = parser.parse_args()
    if args.updates <= 0 or args.updates % 256:
        parser.error('--updates must be a positive multiple of 256')
    assert_work_branch()
    assert_output_boundary(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == 'cpu':
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'
    import flax.serialization
    import jax
    import numpy as np
    from .agents.ibolt import Agent
    from .protocol import configuration

    config = configuration('ibolt', 'medium', seed=0, batch_size=256,
                           updates_per_collect=256)
    rng = np.random.default_rng(1234)
    observations = rng.normal(size=(8192, 4)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(8192, 2)).astype(np.float32)
    next_obs = observations + rng.normal(0, .05, size=(8192, 4)).astype(np.float32)
    done = (rng.random(8192) < .02).astype(np.float32)
    rewards = (100 * (rng.random(8192) < .01)).astype(np.float32)
    agents = {}

    def state(agent):
        model, policy = agent.model, agent.model.policy
        return dict(actor=flax.serialization.to_state_dict(policy.actor_state),
                    critic=flax.serialization.to_state_dict(policy.qf_state),
                    learner_key=model.key, policy_key=policy.key,
                    noise_key=policy.noise_key, updates=np.asarray(agent.updates),
                    timesteps=np.asarray(model.num_timesteps))

    def compare(left, right):
        a, structure = jax.tree_util.tree_flatten(jax.device_get(left))
        b, other = jax.tree_util.tree_flatten(jax.device_get(right))
        if structure != other:
            raise AssertionError('State tree structures differ')
        for index, (x, y) in enumerate(zip(a, b)):
            # Typed JAX keys must be compared through their underlying bits.
            if hasattr(x, 'dtype') and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                x, y = jax.random.key_data(x), jax.random.key_data(y)
            if not np.array_equal(x, y):
                raise AssertionError(f'State leaf {index} differs')
        return len(a)

    try:
        for name, deferred in (('eager', False), ('deferred', True)):
            agent = agents[name] = Agent(config, args.output / name,
                                         device=args.device, defer_metrics=deferred)
            agent.observe(observations, actions, rewards, next_obs, done)
        initial_leaves = compare(state(agents['eager']), state(agents['deferred']))
        times = {}
        for name, agent in agents.items():
            model, policy = agent.model, agent.model.policy
            original = (policy.actor_state, policy.qf_state, model.key,
                        policy.key, policy.noise_key,
                        copy.deepcopy(agent.replay.rng.bit_generator.state))
            def collection(step):
                if name == 'deferred' and args.collection_execution == 'scan':
                    model.num_timesteps = step
                    model.train_collection(batch_size=256, count=256)
                else:
                    for _ in range(256):
                        agent.update(step)
            collection(8448)
            # Compile the transport and metric-flush paths before starting timing.
            agent.flush_logs(8448)
            jax.block_until_ready((policy.actor_state, policy.qf_state))
            policy.actor_state, policy.qf_state, model.key, policy.key, policy.noise_key, replay_rng = original
            agent.replay.rng.bit_generator.state = replay_rng
            model._n_updates = 0
            model._pending_metrics = None
            started = time.perf_counter()
            for i in range(256, args.updates + 1, 256):
                step = 8192 + i
                collection(step)
                if i % 1000 < 256:
                    agent.flush_logs(step)
            jax.block_until_ready((policy.actor_state, policy.qf_state))
            times[name] = time.perf_counter() - started
            print(json.dumps(dict(mode=name, seconds=times[name],
                                  updates_per_second=args.updates / times[name])), flush=True)
        final_leaves = compare(state(agents['eager']), state(agents['deferred']))
        if agents['eager'].replay.rng.bit_generator.state != agents['deferred'].replay.rng.bit_generator.state:
            raise AssertionError('Replay RNG states differ')
        logs = []
        for name in agents:
            path = args.output / name / 'learner/progress.csv'
            with path.open() as stream:
                logs.append(list(csv.DictReader(stream)))
        if logs[0] != logs[1]:
            raise AssertionError('CSV values differ')
        result = dict(updates=args.updates, batch_size=256,
                      collection_execution=args.collection_execution,
                      backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                      initial_equal_leaves=initial_leaves, final_equal_leaves=final_leaves,
                      actor_critic_optimizer_keys_exactly_equal=True,
                      replay_rng_exactly_equal=True, csv_exactly_equal=True,
                      timings_seconds=times,
                      updates_per_second={k: args.updates / v for k, v in times.items()},
                      speedup=times['eager'] / times['deferred'], source_hashes=code_hashes())
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}), flush=True)
    finally:
        for agent in agents.values():
            agent.close()


if __name__ == '__main__':
    main()
