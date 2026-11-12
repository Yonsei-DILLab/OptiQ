"""Train pinned MFPO/MEOW updates with the shared vec256 HER protocol."""
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import subprocess
import time
import traceback


def atomic(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def run(spec, smoke=False):
    source = Path(spec['source'])
    assert subprocess.check_output(['git','branch','--show-current'], cwd=source, text=True).strip() == 'heechan-rebuttal'
    for name, digest in json.loads(Path(spec['source_manifest']).read_text()).items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import jax
    from stable_baselines3.common.vec_env import VecNormalize
    from fetch_rebuttal.env import make_env
    from fetch_rebuttal.run import Baseline, guard
    from fetch_rebuttal.normalization import save_normalizer
    from rebuttal_tools.fetch_suite_vector_env import GroupedFetchVecEnv
    from rebuttal_tools.fetch_suite_vector_replay import VectorHERReplayBuffer, NativeReplayView
    from rebuttal_tools.fetch_suite_evaluation import evaluate
    guard()
    method, seed, task = spec['method'], spec['seed'], spec['task']
    assert method=='mfpo'
    assert (jax.default_backend() == 'gpu') if method == 'mfpo' else torch.cuda.is_available()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    total, warmup = (13312 if smoke else 5000192), 12800
    expected = (total-warmup)//8
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoints','evaluations','learner'):
        (output/name).mkdir()
    config = dict(method=method, task=task, seed=seed, steps=total, target_steps=5000000,
        num_envs=256, batch_size=256, utd=.125, warmup=warmup,
        gradient_steps_per_vector_step=32, replay_capacity=25600 if smoke else 1000000,
        HER=True, normalize_observations=True, norm_reward=False, clip_obs=5.,
        eval_every=5000, eval_episodes=2 if smoke else 10,
        final_episodes=2 if smoke else 100, checkpoint_every=100000,
        expected_updates=expected, smoke=smoke,
        mfpo_target_entropy_coeff=spec['mfpo_target_entropy_coeff'],
        mfpo_initial_temperature=spec['mfpo_initial_temperature'],
        worker_cpus=spec['worker_cpus'], source_manifest=spec['source_manifest'])
    vec = env = agent = None
    started = time.time()
    try:
        vec = VecNormalize(GroupedFetchVecEnv(256, spec['worker_cpus'], task),
            norm_obs=True, norm_reward=False, clip_obs=5., epsilon=1e-8)
        env = make_env(task)
        config.update(observation_dim=vec.observation_space.shape[0], action_dim=4)
        replay = VectorHERReplayBuffer(config['replay_capacity'], vec.observation_space,
            vec.action_space, n_envs=256, device='cpu', compute_reward=env.unwrapped.compute_reward,
            task=task, her_seed=seed, table_height=float(getattr(env.unwrapped, 'height_offset', 0.)))
        agent = Baseline(method, config, output, 'cuda' if method == 'meow' else 'cpu')
        if method == 'mfpo':
            # The older frozen Fetch adapter predates the entropy-grid CLI.
            # Recreate its identical upstream learner from the same seed, changing
            # only the two previously selected MFPO constructor parameters.
            from fetch_rebuttal.baselines.native import SpaceOnlyEnv
            native = agent.native
            previous = dict(native.config)
            native.config.update(temp=spec['mfpo_initial_temperature'],
                target_entropy_coeff=spec['mfpo_target_entropy_coeff'],
                actor_hidden_dims=(256,256),critic_hidden_dims=(256,256),
                actor_layer_norm=False,critic_layer_norm=False)
            assert all(native.config[k] == v for k,v in previous.items()
                if k not in ('temp','target_entropy_coeff','actor_hidden_dims','critic_hidden_dims','actor_layer_norm','critic_layer_norm'))
            descriptor = SpaceOnlyEnv(config['observation_dim'], 4)
            descriptor.observation_space.seed(seed)
            descriptor.action_space.seed(seed)
            training = {k:v for k,v in native.config.items() if k != 'upstream_commit'}
            native.agent = type(native.agent).create(seed, descriptor.observation_space,
                descriptor.action_space, **training)
            evaluation = agent.config['evaluation']
            agent.config = dict(native.config, evaluation=evaluation)
            assert float(native.agent.target_entropy) == spec['mfpo_target_entropy_coeff']*4
        agent.native.replay = NativeReplayView(replay, vec)
        agent.normalizer = vec
        atomic(output/'config.json', config)
        atomic(output/'native-config.json', agent.config)
        if method == 'mfpo':
            assert agent.native.config['temp'] == spec['mfpo_initial_temperature']
            assert agent.native.config['target_entropy_coeff'] == spec['mfpo_target_entropy_coeff']
        def digest():
            if method == 'mfpo':
                import flax.serialization
                return hashlib.sha256(flax.serialization.to_bytes(agent.native.agent.actor.params)).hexdigest()
            return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in agent.native.policy.parameters())).hexdigest()
        from flax.traverse_util import flatten_dict
        actor=agent.native.agent.actor.params['MLP_0']
        critic=agent.native.agent.critic_1.params['MLP_0']
        actor_dense=sorted((k,v['kernel'].shape) for k,v in actor.items() if k.startswith('Dense_'))
        critic_dense=sorted((k,v['kernel'].shape) for k,v in critic.items() if k.startswith('Dense_'))
        assert [list(shape) for _,shape in actor_dense][0][-1]==256
        assert len(actor_dense)==3 and [shape[-1] for _,shape in actor_dense[:-1]]==[256,256]
        assert len(critic_dense)==2 and all(shape[-1]==256 for _,shape in critic_dense)
        norm_paths=[]
        for label,state in [('actor',agent.native.agent.actor),('critic1',agent.native.agent.critic_1),('critic2',agent.native.agent.critic_2),('logp',agent.native.agent.logp_mvel)]:
            norm_paths.extend(label+'/'+str(path) for path in flatten_dict(state.params) if any('layernorm' in str(part).lower() for part in path))
        assert not norm_paths
        assert agent.native.config['actor_layer_norm'] is False and agent.native.config['critic_layer_norm'] is False
        atomic(output/'architecture-audit.json',dict(actor_hidden_dims=[256,256],critic_hidden_dims=[256,256],actor_layer_norm=False,critic_layer_norm=False,actor_dense=actor_dense,critic_dense=critic_dense,layernorm_parameter_paths=norm_paths))
        config.update(actor_hidden_dims=[256,256],critic_hidden_dims=[256,256],actor_layer_norm=False,critic_layer_norm=False)
        atomic(output/'config.json',config)
        initial = digest()
        def save(step, full=False):
            agent.save(output/'checkpoints', step)
            save_normalizer(vec, output/'checkpoints', step)
            if full:
                with (output/'checkpoints/final-replay.pkl').open('wb') as stream:
                    pickle.dump(replay, stream, protocol=pickle.HIGHEST_PROTOCOL)
        def evaluate_at(step):
            count = float(vec.obs_rms.count)
            episodes = config['final_episodes'] if step == total else config['eval_episodes']
            result = evaluate(agent, task, episodes, 700000+seed*1000, 'policy')
            assert float(vec.obs_rms.count) == count
            record = dict(steps=step, updates=agent.updates, primary_mode='policy', policy=result)
            atomic(output/'evaluations'/f'{step:09d}.json', record)
            with (output/'evaluations.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            print(json.dumps(dict(steps=step, success_rate=result['success_rate'])), flush=True)
            return record
        evaluate_at(0)
        vec.seed(seed)
        vec.action_space.seed(seed)
        obs = vec.reset()
        next_eval, next_log, next_checkpoint = 5000, 4096, 100000
        for step in range(256, total+1, 256):
            raw_obs = vec.get_original_obs().copy()
            actions = (np.stack([vec.action_space.sample() for _ in range(256)])
                if step-256 < warmup else agent.act(obs, mode='train'))
            assert actions.shape == (256,4) and np.isfinite(actions).all()
            obs, rewards, dones, infos = vec.step(actions)
            replay.add(raw_obs, vec.get_original_obs(), actions,
                vec.get_original_reward(), dones, infos)
            # Match SB3's evaluation callback: collect, evaluate, then optimize.
            if not smoke and step >= next_eval and step < total:
                evaluate_at(step)
                while next_eval <= step:
                    next_eval += 5000
            metrics = {}
            if step > warmup:
                for _ in range(32):
                    metrics = agent.update(step) or {}
                assert agent.updates == (step-warmup)//8
            if step >= next_log or step == total:
                assert all(math.isfinite(float(v)) for v in metrics.values())
                atomic(output/'progress.json', dict(status='running' if step > warmup else 'warmup',
                    steps=step, updates=agent.updates, utd=.125,
                    elapsed_seconds=time.time()-started, observed_unix=time.time(),
                    metrics=metrics, her_replay_last_sample=replay.last))
                with (output/'learner/metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(steps=step, updates=agent.updates, **metrics))+'\n')
                next_log = step+4096
            if step >= next_checkpoint:
                save(step)
                while next_checkpoint <= step:
                    next_checkpoint += 100000
        assert agent.updates == expected and digest() != initial
        if method == 'mfpo':
            leaves = jax.tree_util.tree_leaves(agent.native.agent)
            assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
            assert int(agent.native.agent.actor.step) == expected
        else:
            assert all(torch.isfinite(p).all().item() for p in agent.native.policy.parameters())
        assert replay.last['relabeled_per_optimizer_minibatch'] == 204
        save(total, full=not smoke)
        final = evaluate_at(total)['policy']
        result = dict(completed=True, method=method, task=task, seed=seed,
            steps=total, updates=expected, observed_utd=.125, smoke=smoke,
            elapsed_seconds=time.time()-started, finished_unix=time.time(), backend='gpu',
            final_success=final['success_rate'], actor_parameters_changed=True,
            native_learner_unchanged=True, her=replay.last, final_stages=final.get('stages',{}))
        atomic(output/'verified-completion.json', result)
        atomic(output/'progress.json', dict(result, status='completed'))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        atomic(output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), observed_unix=time.time()))
        raise
    finally:
        if vec is not None:
            vec.close()
        if env is not None:
            env.close()
        if agent is not None:
            agent.close()


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(json.loads(args.spec.read_text()),args.smoke)


if __name__=='__main__':main()
