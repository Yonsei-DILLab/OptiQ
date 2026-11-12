"""Adroit collector: a few processes, each stepping a group of native environments."""
import multiprocessing as mp
import os
import traceback
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def _worker(remote, parent, count, cpus, task):
    parent.close()
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    if cpus:
        os.sched_setaffinity(0, cpus)
    from adroit_online.env import make_env
    envs = []
    try:
        envs = [make_env(task) for _ in range(count)]
        remote.send(('ok', (envs[0].observation_space, envs[0].action_space)))
        def attr(env, name):
            return env.get_wrapper_attr(name)
        while True:
            command, data = remote.recv()
            if command == 'reset':
                records = [e.reset(seed=seed) for e, seed in zip(envs, data)]
                answer = (np.stack([x[0] for x in records]), [x[1] for x in records])
            elif command == 'step':
                observations, rewards, dones, infos, resets = [], [], [], [], []
                for env, action in zip(envs, data):
                    obs, reward, terminated, truncated, info = env.step(action)
                    info = dict(info)
                    done = terminated or truncated
                    info['TimeLimit.truncated'] = truncated and not terminated
                    reset_info = None
                    if done:
                        info['terminal_observation'] = obs.copy()
                        obs, reset_info = env.reset()
                    observations.append(obs);rewards.append(reward);dones.append(done)
                    infos.append(info);resets.append(reset_info)
                answer = (np.stack(observations), np.asarray(rewards, np.float32),
                          np.asarray(dones, bool), infos, resets)
            elif command == 'get_attr':
                name, indices = data
                answer = [attr(envs[i], name) for i in indices]
            elif command == 'set_attr':
                name, value, indices = data
                for i in indices:
                    envs[i].set_wrapper_attr(name, value)
                answer = None
            elif command == 'env_method':
                name, args, kwargs, indices = data
                answer = [attr(envs[i], name)(*args, **kwargs) for i in indices]
            elif command == 'is_wrapped':
                from stable_baselines3.common.env_util import is_wrapped
                cls, indices = data
                answer = [is_wrapped(envs[i], cls) for i in indices]
            elif command == 'close':
                remote.send(('ok', None))
                break
            else:
                raise ValueError(command)
            remote.send(('ok', answer))
    except EOFError:
        pass
    except BaseException:
        try:
            remote.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        for env in envs:
            env.close()
        remote.close()


class GroupedAdroitVecEnv(VecEnv):
    def __init__(self, task, num_envs=256, worker_cpus=None):
        worker_cpus = worker_cpus or [[] for _ in range(4)]
        assert num_envs % len(worker_cpus) == 0
        self.group_size = num_envs // len(worker_cpus)
        self.remotes, self.processes = [], []
        self.waiting = self.closed = False
        context = mp.get_context('spawn')
        for cpus in worker_cpus:
            remote, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, remote, self.group_size, cpus, task), daemon=True)
            process.start();child.close()
            self.remotes.append(remote);self.processes.append(process)
        spaces = [self._receive(r) for r in self.remotes]
        super().__init__(num_envs, *spaces[0])

    @staticmethod
    def _receive(remote):
        if not remote.poll(180):
            raise TimeoutError('Adroit environment worker did not respond')
        status, value = remote.recv()
        if status != 'ok':
            raise RuntimeError(value)
        return value

    def reset(self):
        for k, remote in enumerate(self.remotes):
            remote.send(('reset', self._seeds[k*self.group_size:(k+1)*self.group_size]))
        values = [self._receive(r) for r in self.remotes]
        self.reset_infos = [x for _, infos in values for x in infos]
        self._reset_seeds();self._reset_options()
        return np.concatenate([x[0] for x in values])

    def step_async(self, actions):
        for remote, batch in zip(self.remotes, np.split(actions, len(self.remotes))):
            remote.send(('step', batch))
        self.waiting = True

    def step_wait(self):
        values = [self._receive(r) for r in self.remotes]
        self.waiting = False
        reset_infos = [x for v in values for x in v[4]]
        for i, reset in enumerate(reset_infos):
            if reset is not None:
                self.reset_infos[i] = reset
        return (np.concatenate([v[0] for v in values]), np.concatenate([v[1] for v in values]),
                np.concatenate([v[2] for v in values]), [x for v in values for x in v[3]])

    def _selected(self, command, indices, make_data):
        indices = list(self._get_indices(indices));answers = {}
        for k, remote in enumerate(self.remotes):
            local = [i-k*self.group_size for i in indices if k*self.group_size <= i < (k+1)*self.group_size]
            if local:
                remote.send((command, make_data(local)))
                answers[k] = local
        result = {}
        for k, local in answers.items():
            values = self._receive(self.remotes[k])
            if values is not None:
                result.update({k*self.group_size+i: v for i, v in zip(local, values)})
        return [result[i] for i in indices] if result else None

    def get_attr(self, attr_name, indices=None):
        return self._selected('get_attr', indices, lambda ids: (attr_name, ids))

    def set_attr(self, attr_name, value, indices=None):
        self._selected('set_attr', indices, lambda ids: (attr_name, value, ids))

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._selected('env_method', indices, lambda ids: (method_name, method_args, method_kwargs, ids))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._selected('is_wrapped', indices, lambda ids: (wrapper_class, ids))

    def get_images(self):
        return [None] * self.num_envs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    self._receive(remote)
                except (EOFError, RuntimeError, TimeoutError):
                    pass
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate();process.join(timeout=5)
        for remote in self.remotes:
            remote.close()
        self.closed = True
