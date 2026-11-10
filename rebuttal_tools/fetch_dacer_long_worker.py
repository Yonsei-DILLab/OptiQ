"""Fixed-budget DACER study; unchanged learners with matched 5M/10M evaluations."""
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
    import jax
    from omegaconf import OmegaConf
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize
    from fetch_rebuttal.env import make_env
    from fetch_rebuttal.normalization import normalized_ibolt_class, save_normalizer
    from fetch_rebuttal.run import IBOLTView, guard
    from rebuttal_tools.fetch_vector_env import GroupedFetchVecEnv
    from rebuttal_tools.fetch_vector_replay import VectorHERReplayBuffer
    from rebuttal_tools.fetch_grasp_evaluation import evaluate
    guard()
    method, seed = spec['method'], spec['seed']
    assert method in ('sac', 'ibolt')
    if method == 'sac': assert torch.cuda.is_available()
    if method == 'ibolt': assert jax.default_backend() == 'gpu'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n_envs, warmup, gradients = 256, 12800, 32
    total = 13568 if smoke else 10000128
    milestones = {13312} if smoke else {5000192}
    target = float(spec.get("target_entropy_per_dim", -.9))
    scale = float(spec.get("noise_scale", 2.))
    assert -.9 <= target <= .75 and .1 <= scale <= 3.
    expected = (total - warmup) // 8
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoints', 'evaluations', 'learner'): (output/name).mkdir()
    settings = dict(method=method, task='pick-and-place', seed=seed, steps=total, target_steps=10000000,
        warmup=warmup, batch_size=256, num_envs=n_envs, utd=.125, gradient_steps_per_vector_step=gradients,
        replay_capacity=25600 if smoke else 1000000, eval_every=5000,
        eval_episodes=2 if smoke else 10, final_episodes=2 if smoke else 500, milestone_episodes=2 if smoke else 500,
        milestone_steps=sorted(milestones), profile=spec["profile"],
        checkpoint_every=100000, HER=True, normalize_observations=True,
        expected_updates=expected, smoke=smoke, worker_cpus=spec['worker_cpus'], source_manifest=spec['source_manifest'],
        primary_success='Final-step success; SAC deterministic actor; iBOLT fresh stochastic z and conditional mean only')
    if method == 'ibolt':
        settings.update(temperature=.25, log_std_min=-5., log_std_max=1., initial_log_std=-1.,
            dacer_target_entropy_per_dim=target, dacer_noise_scale=scale,
            parameter_selection='Prespecified two-seed 10M search; additional three seeds after local selection')
    atomic(output/'config.json', settings)
    atomic(output/'progress.json', dict(status='initializing', steps=0, updates=0, observed_unix=time.time()))
    model = vec = reward_env = None; started = time.time()
    try:
        vec = VecNormalize(GroupedFetchVecEnv(n_envs, spec['worker_cpus']), norm_obs=True,
            norm_reward=False, clip_obs=5., epsilon=1e-8)
        reward_env = make_env('pick-and-place')
        common = dict(learning_starts=warmup, batch_size=256, buffer_size=settings['replay_capacity'],
            train_freq=1, gradient_steps=gradients, replay_buffer_class=VectorHERReplayBuffer,
            replay_buffer_kwargs=dict(compute_reward=reward_env.unwrapped.compute_reward, her_seed=seed,
                table_height=float(reward_env.unwrapped.height_offset)))
        if method == 'ibolt':
            import train
            from exploration import ExplorationIBOLT
            cfg = train.compose_config(['benchmark=ant', 'wandb.activate=false'])
            cfg.env_name='FetchPickAndPlace-v4'; cfg.task='pick-and-place'; cfg.seed=seed; cfg.temperature=.25
            cfg.total_steps=total; cfg.output_root=str(output); cfg.require_gpu=True
            cfg.alg.learning_starts=cfg.alg.actor.learning_starts=warmup
            cfg.alg.batch_size=256; cfg.alg.buffer_size=settings['replay_capacity']; cfg.alg.utd=.125
            train.validate_config(cfg)
            cfg.dacer.noise_scale=scale; cfg.dacer.target_entropy_per_dim=target
            cfg.alg.actor.log_std_min=-5.; cfg.alg.actor.log_std_max=1.; cfg.alg.actor.initial_log_std=-1.
            assert list(cfg.alg.actor.hidden_dims)==[256,256] and list(cfg.alg.critic.hs)==[256,256]
            assert cfg.alg.actor.num_policy_samples==64 and cfg.alg.actor.proposals_per_policy_sample==1
            atomic(output/'mujoco-config.json', OmegaConf.to_container(cfg, resolve=True))
            # All algorithm-specific hyperparameters are frozen to the existing winning profile.
            reference=json.loads(Path(spec['ibolt_reference_config']).read_text())
            def leaves(value, prefix=''):
                if isinstance(value, dict):
                    for key, item in value.items(): yield from leaves(item, prefix+'.'+key if prefix else key)
                else: yield prefix, value
            actual=dict(leaves(OmegaConf.to_container(cfg, resolve=True))); old=dict(leaves(reference))
            allowed={'seed','run_name','total_steps','output_root','alg.learning_starts','alg.actor.learning_starts','alg.utd','alg.buffer_size','dacer.noise_scale','dacer.target_entropy_per_dim'}
            assert set(actual)==set(old)
            assert all(actual[k]==old[k] for k in actual if k not in allowed), {k:(old[k],actual[k]) for k in actual if k not in allowed and actual[k]!=old[k]}
            cls=normalized_ibolt_class(ExplorationIBOLT)
            # iBOLT obtains these common values from cfg; avoid duplicate constructor arguments.
            ibolt_common={k:v for k,v in common.items() if k not in ('learning_starts','batch_size','buffer_size','train_freq')}
            model=cls('MlpPolicy', env=vec, cfg=cfg, model_save_path=str(output/'checkpoints'),
                save_every_n_steps=100000, **ibolt_common)
            view=IBOLTView(model)
            import flax.serialization
            initial_digest=hashlib.sha256(flax.serialization.to_bytes(model.policy.actor_state.params)).hexdigest()
        else:
            class RawSAC(SAC):
                def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
                    super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
                    index=(replay_buffer.pos-1)%replay_buffer.buffer_size
                    for i, info in enumerate(infos):
                        replay_buffer.next_observations[index,i]=info['fetch_raw_next_observation']
            model=RawSAC('MlpPolicy', env=vec, learning_rate=3e-4, tau=.005, gamma=.99,
                ent_coef='auto', policy_kwargs=dict(net_arch=[256,256]), seed=seed, device='cuda', verbose=0, **common)
            if smoke:
                from fetch_rebuttal.baselines.native import SAC as NativeSAC
                reference_folder=output/'native-reference'; (reference_folder/'learner').mkdir(parents=True)
                py_state, np_state=random.getstate(), np.random.get_state()
                try:
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        reference=NativeSAC(seed, reference_folder, total, 28, batch_size=256, device='cuda')
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
                actor_lr=3e-4, critic_lr=3e-4, entropy='automatic', gamma=.99, tau=.005,
                target_update_interval=1, target_entropy='auto', **settings))
        model.set_logger(configure(str(output/'learner'), ['csv']))
        def save():
            if method=='ibolt': model._save_model()
            else:
                model.save(output/'checkpoints'/f'policy_{model.num_timesteps:09d}.zip', exclude=['train'])
                save_normalizer(vec, output/'checkpoints', model.num_timesteps)
        def evaluate_at(step):
            count=float(vec.obs_rms.count)
            episodes=(settings['final_episodes'] if step==total else settings['milestone_episodes'] if step in milestones else settings['eval_episodes'])
            modes=('stochastic_z','zero_z') if method=='ibolt' else ('policy',)
            record=dict(steps=step, updates=model._n_updates, primary_mode=modes[0])
            for mode in modes:
                record[mode]=evaluate(view, episodes, 700000+seed*1000, mode)
                if smoke:
                    from fetch_rebuttal.run import evaluate as original_evaluate
                    original=original_evaluate(view,'pick-and-place',episodes,700000+seed*1000,mode)
                    assert all(record[mode][k]==v for k,v in original.items()), 'Stage instrumentation changed original evaluation'
            assert float(vec.obs_rms.count)==count, 'Evaluation altered normalization statistics'
            if step in milestones or step==total:
                for filename in ['evaluations.jsonl','height-evaluations.jsonl']:
                    f=output/filename
                    if f.exists():f.write_text(''.join(line+'\n' for line in f.read_text().splitlines() if json.loads(line)['steps']!=step))
            atomic(output/'evaluations'/f'{step:09d}.json', record)
            with (output/'evaluations.jsonl').open('a') as f: f.write(json.dumps(record, allow_nan=False)+'\n')
            heights=dict(steps=step,modes={m:record[m]['stages'] for m in modes})
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
            if method=='ibolt': assert int(model.policy.actor_state.step)==model._n_updates
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
            if step in milestones:
                save()
                milestone=evaluate_at(step)
                atomic(output/f'milestone-{step:09d}.json',milestone)
                assert milestone['updates']==(step-warmup)//8
        model.train=measured_train
        class Evaluation(BaseCallback):
            def __init__(self): super().__init__(); self.next_eval=5000
            def _on_step(self):
                step=int(self.num_timesteps)
                if step%4096==0 and step<=warmup:
                    atomic(output/'progress.json',dict(status='warmup',steps=step,updates=0,elapsed_seconds=time.time()-started,observed_unix=time.time()))
                if not smoke and step>=self.next_eval and step<total:
                    if step not in milestones:evaluate_at(step)
                    while self.next_eval<=step: self.next_eval+=5000
                return True
        evaluate_at(0)
        model.learn(total_timesteps=total, callback=Evaluation(), log_interval=100000000)
        assert model.num_timesteps==total and model._n_updates==expected
        if method=='ibolt':
            import flax.serialization
            final_digest=hashlib.sha256(flax.serialization.to_bytes(model.policy.actor_state.params)).hexdigest()
            for state in (model.policy.actor_state, model.policy.qf_state):
                assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
            assert int(model.policy.actor_state.step)==expected
        else:
            assert all(torch.isfinite(p).all().item() for p in model.policy.parameters())
            final_digest=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.actor.parameters())).hexdigest()
        assert final_digest!=initial_digest, 'Actor parameters did not change'
        save()
        if not smoke: model.save_replay_buffer(output/'checkpoints/final-replay.pkl')
        final=evaluate_at(total); primary='stochastic_z' if method=='ibolt' else 'policy'
        result=dict(completed=True,method=method,profile=spec['profile'],seed=seed,steps=total,updates=expected,observed_utd=.125,
            elapsed_seconds=time.time()-started,finished_unix=time.time(),backend='gpu',smoke=smoke,
            final_success=final[primary]['success_rate'],actor_parameters_changed=True,her=model.replay_buffer.last,
            final_stages=final[primary]['stages'],target_entropy_per_dim=target if method=='ibolt' else None,
            noise_scale=scale if method=='ibolt' else None,milestones=sorted(milestones))
        for step in milestones:
            assert (output/f'milestone-{step:09d}.json').exists()
        if method=='ibolt':
            reg=json.loads((output/'dacer_regulator.json').read_text())
            assert all(math.isfinite(reg[k]) for k in ('alpha','noise_std','entropy_proxy'))
            assert abs(reg['noise_std']-scale*reg['alpha'])<1e-6
            if smoke:
                error=reg['entropy_proxy']-4*target
                if error>1e-6:assert reg['alpha']<.27
                if error< -1e-6:assert reg['alpha']>.27
            result['regulator']=reg
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
    args=p.parse_args();run(json.loads(args.spec.read_text()),args.smoke)


if __name__=='__main__':main()
