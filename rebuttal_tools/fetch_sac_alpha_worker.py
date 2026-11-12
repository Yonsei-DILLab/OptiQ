"""Fixed-alpha SAC Push sweep; shared vec256 HER collection and UTD1/8."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


def atomic(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def run(spec, smoke=False):
    source = Path(spec['source']).resolve(); sys.path.insert(0, str(source))
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=source, text=True).strip() == 'heechan-rebuttal'
    for name, digest in json.loads(Path(spec['source_manifest']).read_text()).items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize
    from fetch_rebuttal.env import make_env
    from fetch_rebuttal.normalization import save_normalizer
    from fetch_rebuttal.run import guard
    from rebuttal_tools.fetch_suite_vector_env import GroupedFetchVecEnv
    from rebuttal_tools.fetch_suite_vector_replay import VectorHERReplayBuffer
    from rebuttal_tools.fetch_suite_evaluation import evaluate
    guard()
    method, seed, task = spec['method'], spec['seed'], spec['task']
    alpha = float(spec['alpha'])
    assert method == 'sac' and task == 'push' and math.isfinite(alpha) and alpha>0
    assert torch.cuda.is_available()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n_envs, warmup, gradients = 256, 12800, 32
    total = 13312 if smoke else 5000192
    expected = (total - warmup) // 8
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoints', 'evaluations', 'learner'): (output/name).mkdir()
    settings = dict(method=method, task=task, seed=seed, steps=total, target_steps=5000000,
        warmup=warmup, batch_size=256, num_envs=n_envs, utd=.125, gradient_steps_per_vector_step=gradients,
        replay_capacity=25600 if smoke else 1000000, eval_every=5000,
        eval_episodes=2 if smoke else 10, final_episodes=2 if smoke else 100,
        checkpoint_every=100000, HER=True, normalize_observations=True,
        expected_updates=expected, smoke=smoke, worker_cpus=spec['worker_cpus'], source_manifest=spec['source_manifest'],
        primary_success='Final-step success; SAC deterministic actor')
    settings.update(entropy_mode='fixed',alpha=alpha,alpha_optimizer=False)
    atomic(output/'config.json', settings)
    atomic(output/'progress.json', dict(status='initializing', steps=0, updates=0, observed_unix=time.time()))
    model = vec = reward_env = None; started = time.time()
    try:
        vec = VecNormalize(GroupedFetchVecEnv(n_envs, spec['worker_cpus'], task=task), norm_obs=True,
            norm_reward=False, clip_obs=5., epsilon=1e-8)
        reward_env = make_env(task)
        common = dict(learning_starts=warmup, batch_size=256, buffer_size=settings['replay_capacity'],
            train_freq=1, gradient_steps=gradients, replay_buffer_class=VectorHERReplayBuffer,
            replay_buffer_kwargs=dict(compute_reward=reward_env.unwrapped.compute_reward, her_seed=seed,
                table_height=float(getattr(reward_env.unwrapped, 'height_offset', 0.)), task=task))
        class RawSAC(SAC):
            def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
                super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
                index=(replay_buffer.pos-1)%replay_buffer.buffer_size
                for i, info in enumerate(infos):
                    replay_buffer.next_observations[index,i]=info['fetch_raw_next_observation']
        model=RawSAC('MlpPolicy', env=vec, learning_rate=3e-4, tau=.005, gamma=.99,
            ent_coef=alpha, policy_kwargs=dict(net_arch=[256,256]), seed=seed, device='cuda', verbose=0, **common)
        assert model.ent_coef_optimizer is None and model.log_ent_coef is None
        assert math.isclose(float(model.ent_coef_tensor.item()),alpha,rel_tol=1e-6)
        if smoke:
            from fetch_rebuttal.baselines.native import SAC as NativeSAC
            reference_folder=output/'native-reference'; (reference_folder/'learner').mkdir(parents=True)
            py_state, np_state=random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    reference=NativeSAC(seed, reference_folder, total, vec.observation_space.shape[0], batch_size=256, device='cuda')
                    old=reference.model.policy.state_dict(); actual=model.policy.state_dict()
                    assert set(old)==set(actual)
                    assert all(torch.equal(actual[k],old[k]) for k in old), 'SAC native initialization differs'
                    assert model.gamma==reference.model.gamma and model.tau==reference.model.tau
                    assert model.target_entropy==reference.model.target_entropy
                    reference.model.logger.close(); del reference
                atomic(output/'native-sac-verification.json', dict(passed=True,
                    exact_initial_policy_and_critic_tensors=True, matching_gamma_tau_target_entropy=True))
            finally:
                random.setstate(py_state); np.random.set_state(np_state)
        class SACView:
            def __init__(self, model): self.model=model
            def act(self, obs, mode='policy'):
                obs=self.model.get_vec_normalize_env().normalize_obs(obs)
                return self.model.predict(obs, deterministic=True)[0]
            @contextmanager
            def evaluation_rng(self, eval_seed):
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(eval_seed); torch.cuda.manual_seed_all(eval_seed); yield
        view=SACView(model)
        initial_digest=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.actor.parameters())).hexdigest()
        atomic(output/'sac-config.json', dict(algorithm='Stable-Baselines3 SAC', network=[256,256],
            actor_lr=3e-4, critic_lr=3e-4, entropy='fixed', alpha_value=alpha, gamma=.99, tau=.005,
            target_update_interval=1, target_entropy='auto', target_entropy_used=False, **settings))
        model.set_logger(configure(str(output/'learner'), ['csv']))
        def save():
            model.save(output/'checkpoints'/f'policy_{model.num_timesteps:09d}.zip', exclude=['train'])
            save_normalizer(vec, output/'checkpoints', model.num_timesteps)
        def evaluate_at(step):
            count=float(vec.obs_rms.count)
            episodes=settings['final_episodes'] if step==total else settings['eval_episodes']
            modes=('policy',)
            record=dict(steps=step, updates=model._n_updates, primary_mode=modes[0])
            for mode in modes:
                record[mode]=evaluate(view, task, episodes, 700000+seed*1000, mode)
                if smoke:
                    from fetch_rebuttal.run import evaluate as original_evaluate
                    original=original_evaluate(view,task,episodes,700000+seed*1000,mode)
                    assert all(record[mode][k]==v for k,v in original.items()), 'Stage instrumentation changed original evaluation'
            assert float(vec.obs_rms.count)==count, 'Evaluation altered normalization statistics'
            atomic(output/'evaluations'/f'{step:09d}.json', record)
            with (output/'evaluations.jsonl').open('a') as f: f.write(json.dumps(record, allow_nan=False)+'\n')
            heights=dict(steps=step,modes={m:record[m].get('stages',{}) for m in modes})
            with (output/'height-evaluations.jsonl').open('a') as f: f.write(json.dumps(heights, allow_nan=False)+'\n')
            print(json.dumps(dict(steps=step, success_rate=record[modes[0]]['success_rate'])), flush=True)
            return record
        next_log=warmup; next_checkpoint=100000
        original_train=model.train
        def measured_train(batch_size, gradient_steps):
            nonlocal next_log, next_checkpoint
            assert batch_size==256 and gradient_steps==32
            original_train(batch_size=batch_size, gradient_steps=gradient_steps)
            step=int(model.num_timesteps)
            assert model._n_updates==(step-warmup)//8
            assert model.ent_coef_optimizer is None
            assert math.isclose(float(model.ent_coef_tensor.item()),alpha,rel_tol=1e-6)
            metrics={}
            for k,v in model.logger.name_to_value.items():
                if isinstance(v,(int,float,np.number)):
                    value=float(v)
                    if k.startswith('train/') and not math.isfinite(value): raise FloatingPointError(k)
                    if math.isfinite(value): metrics[k]=value
            if step>=next_log or step==total:
                atomic(output/'progress.json', dict(status='running', steps=step, updates=model._n_updates,
                    elapsed_seconds=time.time()-started, observed_unix=time.time(), utd=.125, metrics=metrics,
                    her_replay_last_sample=model.replay_buffer.last))
                model.logger.dump(step); next_log=step+4096
            if step>=next_checkpoint:
                save()
                while next_checkpoint<=step: next_checkpoint+=100000
        model.train=measured_train
        class Evaluation(BaseCallback):
            def __init__(self): super().__init__(); self.next_eval=5000
            def _on_step(self):
                step=int(self.num_timesteps)
                if step%4096==0 and step<=warmup:
                    atomic(output/'progress.json',dict(status='warmup',steps=step,updates=0,elapsed_seconds=time.time()-started,observed_unix=time.time()))
                if not smoke and step>=self.next_eval and step<total:
                    evaluate_at(step)
                    while self.next_eval<=step: self.next_eval+=5000
                return True
        evaluate_at(0)
        model.learn(total_timesteps=total, callback=Evaluation(), log_interval=100000000)
        assert model.num_timesteps==total and model._n_updates==expected
        assert all(torch.isfinite(p).all().item() for p in model.policy.parameters())
        final_digest=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.actor.parameters())).hexdigest()
        assert final_digest!=initial_digest, 'Actor parameters did not change'
        save()
        if not smoke: model.save_replay_buffer(output/'checkpoints/final-replay.pkl')
        final=evaluate_at(total); primary='policy'
        result=dict(completed=True,method=method,seed=seed,steps=total,updates=expected,observed_utd=.125,
            elapsed_seconds=time.time()-started,finished_unix=time.time(),backend='gpu',smoke=smoke,
            final_success=final[primary]['success_rate'],actor_parameters_changed=True,her=model.replay_buffer.last,
            final_stages=final[primary].get('stages',{}),task=task,alpha=alpha,entropy_mode='fixed')
        atomic(output/'verified-completion.json',result);atomic(output/'progress.json',dict(result,status='completed'))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        atomic(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),observed_unix=time.time()))
        raise
    finally:
        if vec is not None: vec.close()
        if reward_env is not None: reward_env.close()
        if model is not None: model.logger.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();spec=json.loads(args.spec.read_text())
    run(spec,args.smoke)


if __name__=='__main__':main()
