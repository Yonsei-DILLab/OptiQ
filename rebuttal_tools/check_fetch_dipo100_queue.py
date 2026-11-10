"""CPU-only reservation checks; no DiPo training starts before dependencies finish."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np


def check_gate():
    from rebuttal_tools.fetch_dipo100_gate import inspect_dependency, collect_gate
    tasks = ['pick-and-place','push','slide','reach']
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        jobs = []
        for method in ['ibolt','sac']:
            for task in tasks:
                for seed in range(5):
                    out = root/f'{method}-{task}-{seed}'
                    (out/'evaluations').mkdir(parents=True)
                    cfg = dict(task=task,num_envs=256,batch_size=256,utd=.125,HER=True,normalize_observations=True)
                    done = dict(completed=True,smoke=False,method=method,seed=seed,steps=5000192,updates=623424,observed_utd=.125,finished_unix=1.)
                    (out/'config.json').write_text(json.dumps(cfg))
                    (out/'verified-completion.json').write_text(json.dumps(done))
                    (out/'evaluations/005000192.json').write_text(json.dumps(dict(primary_mode='policy',policy=dict(episodes=100))))
                    jobs.append(dict(method=method,task=task,seed=seed,output=str(out),status='completed'))
        plan_path = root/'plan.json'
        plan_path.write_text(json.dumps(dict(jobs=jobs)))
        spec = dict(plan=str(plan_path),methods=['ibolt','sac'],tasks=tasks)
        assert inspect_dependency(spec)['ready']
        marker = Path(jobs[-1]['output'])/'verified-completion.json'
        saved = marker.read_text()
        marker.unlink()
        result = inspect_dependency(spec)
        assert not result['ready'] and result['completed']==39
        wrong = json.loads(saved)
        wrong['steps'] = 1000000
        marker.write_text(json.dumps(wrong))
        try:
            inspect_dependency(spec)
        except AssertionError:
            pass
        else:
            raise AssertionError('Wrong training budget released the queue')
        marker.write_text(saved)
        sac = dict(id='sac',local=True,plan=str(plan_path),methods=['sac'],tasks=tasks[1:],expected=15)
        pp = dict(id='pickplace',local=True,plan=str(plan_path),methods=['ibolt','sac'],tasks=tasks[:1],expected=10)
        ibolt = dict(id='ibolt',plan=str(plan_path),methods=['ibolt'],tasks=tasks[1:],expected=15,
            host='example.invalid',port=22,user='test')
        plan = dict(dependencies=[sac,pp,ibolt],reader_private_key='unused',reader_known_hosts='unused')
        response = SimpleNamespace(returncode=0,stdout=json.dumps(inspect_dependency(ibolt)),stderr='')
        with patch('subprocess.run',return_value=response):
            assert collect_gate(plan)['ready']
        with patch('subprocess.run',side_effect=TimeoutError('offline')):
            result = collect_gate(plan)
            assert not result['ready'] and result['errors']
    return dict(passed=True,checks=['Require all 40 valid completions',
        '39/40 remains waiting','Reject a different budget','Remote read failure never opens the gate'])


def check_adapter(task):
    import torch
    from gymnasium import spaces
    from fetch_rebuttal.dipo import Agent
    from rebuttal_tools.fetch_dipo100_adapter import DIPOReplayBuffer,bind_replay,match_hidden_layers
    dim = 13 if task=='reach' else 28
    offset = 0 if task=='reach' else 3
    def reward(achieved,desired,infos):
        return -(np.linalg.norm(achieved-desired,axis=-1)>.05).astype(np.float32)
    replay = DIPOReplayBuffer(28,spaces.Box(-np.inf,np.inf,(dim,),np.float32),
        spaces.Box(-1,1,(4,),np.float32),n_envs=2,device='cpu',task=task,
        compute_reward=reward,her_seed=41)
    def state(t):
        x = np.zeros((2,dim),np.float32)
        x[:,offset] = np.arange(2)*100+t
        x[:,offset+1] = t
        x[:,offset+2] = .4
        x[:,-3:] = -99
        return x
    for t in range(40):
        nxt = state(t+1)
        done = np.array([(t+1)%3==0,(t+1)%4==0])
        infos = [dict(fetch_raw_next_observation=nxt[i],**{'TimeLimit.truncated':bool(done[i])}) for i in range(2)]
        replay.add(state(t),nxt,np.zeros((2,4)),np.full(2,-1.),done,infos)
    class Normalizer:
        def normalize_obs(self,x):return x+1000
        def normalize_reward(self,r):return r
    with tempfile.TemporaryDirectory() as folder:
        native = Agent(dict(seed=17,batch_size=256,replay_capacity=256,observation_dim=dim,action_dim=4),Path(folder),device='cpu')
        match_hidden_layers(native)
        bind_replay(native,replay,Normalizer())
        assert native.agent.actor.n_timesteps==native.agent.actor_target.n_timesteps==100
        assert native.agent.action_gradient_steps==20
        s,a,r,n,m = native.agent.memory.sample(256)
        assert s.shape==(256,dim) and torch.all(m==.99)
        assert (s[:,-3]>=1000).sum().item()==204
        states,actions,indices = native.agent.diffusion_memory.sample(256)
        assert actions.requires_grad and (states[:,-3]>=1000).sum().item()==204
        before = replay.actions.copy()
        native.agent.diffusion_memory.replace(indices,np.ones((256,4),np.float32))
        assert np.array_equal(before,replay.actions), 'Changed upstream nonpersistent replacement'
        calls = []
        handle = native.agent.actor.model.register_forward_hook(lambda *args:calls.append(1))
        try:
            with native.evaluation_rng(91):
                action = native.act(np.zeros((1,dim),np.float32),mode='policy')
        finally:
            handle.remove()
        assert len(calls)==100 and np.isfinite(action).all()
        assert native.updates==native.agent.step==0, 'Reservation validation trained DiPo early'
        native.assert_finite()
    return dict(passed=True,task=task,diffusion_steps=100,learner_updates=0,
        actor_hidden_dims=[256,256],critic_hidden_dims=[256,256],layer_norm=False,
        checks=['Native 100 reverse steps observed','HER in actor and critic batches',
            'Normalized goal input','Gamma and timeout masks','Original action-bank replacement preserved'])


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(dict(gate=check_gate(),adapters=[check_adapter(t) for t in ['pick-and-place','push','slide','reach']])),flush=True)
