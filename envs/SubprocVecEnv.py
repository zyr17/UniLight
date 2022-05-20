import torch
import multiprocessing
import numpy as np
import time
import os
from envs.CityFlowEnv import CityFlowEnv
from utils.utils import flatten_data
from utils.log import *


class EnvWorker(multiprocessing.Process):
    def __init__(self, env, envargs, pipe1, pipe2, index = -1):
        multiprocessing.Process.__init__(self, daemon = True)
        self.pipe = pipe1
        self.pipe2 = pipe2
        self.envargs = envargs
        self.env = env
        self.index = index

    def run(self):
        self.env = self.env(*self.envargs)
        self.pipe2.close()
        while True:
            try:
                cmd, data = self.pipe.recv()
                # print(self.index, 'Worker', cmd)
                if cmd == 'step':
                    self.pipe.send(self.env.step(data))
                elif cmd == 'close':
                    self.pipe.close()
                    break
                elif cmd == 'reset':
                    self.pipe.send(self.env.reset())
                elif cmd == 'observation_space':
                    self.pipe.send(self.env.observation_space)
                elif cmd == 'action_space':
                    self.pipe.send(self.env.action_space)
                elif cmd == 'replay_count':
                    if data is not None:
                        assert isinstance(data, int)
                        self.env.replay_count = data
                    self.pipe.send(self.env.replay_count)
                elif cmd == 'get_default_action':
                    self.pipe.send(self.env.get_default_action())
                else:
                    raise NotImplementedError
            except EOFError:
                self.pipe.close()
                break
            except IndexError:
                self.pipe.close()
                break
            except KeyboardInterrupt:
                self.pipe.close()
                break
            except Exception as e:
                log('Unknown exception!', e)
                self.pipe.close()
                break


class EnvVecs:
    def __init__(self, env_class, n_envs, env_args, arg_seed_pos = -1, 
                 seed = 0, is_args_list = False, stagger = False,
                 auto_reset = False):
        self.waiting = False
        self.closed = False

        if not is_args_list:
            args = []
            for _ in range(n_envs):
                args.append(list(env_args))
            if arg_seed_pos != -1:
                for i in range(n_envs):
                    args[i][arg_seed_pos] = seed + i
            env_args = args
        else:
            assert(len(env_args) == n_envs)
            args = []
            for a in env_args:
                args.append(list(a))
            if arg_seed_pos != -1:
                log('set seed pos when use args list, '
                    'will change seed in args list.', level = 'WARN')
                for i in range(n_envs):
                    args[i][arg_seed_pos] = seed + i
            env_args = args

        self.remotes, self.work_remotes = zip(
            *[multiprocessing.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for num, [args, work_remote, remote] in enumerate(zip(
                                                    env_args, 
                                                    self.work_remotes, 
                                                    self.remotes)):
            # args = (env_class, work_remote, remote)
            # daemon=True: if the main process crashes, 
            #              we should not cause things to hang
            process = EnvWorker(env_class, args, work_remote, remote, num)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.is_reset = False
        self.need_stagger = stagger
        self.auto_reset = auto_reset

        self._set_observation_space()
        self._set_action_space()

    def _set_observation_space(self):
        self.remotes[0].send(('observation_space', None))
        self.observation_space = self.remotes[0].recv()

    def _set_action_space(self):
        self.remotes[0].send(('action_space', None))
        self.action_space = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, ists, infos = zip(*results)
        obs = list(obs)
        infos = list(infos)
        # very ugly auto reset implementation
        if self.auto_reset:
            for num, remote in enumerate(self.remotes):
                if ists[num]:
                    remote.send(('reset', None))
                    obs[num], _ = remote.recv()
        return obs, np.stack(rews), np.stack(ists), infos

    def step(self, actions):
        assert(self.is_reset)
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def reset(self, **kwargs):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        if self.need_stagger:
            results = self.stagger(**kwargs, prev_results = results)
        obs, infos = zip(*results)
        self.is_reset = True
        return obs, infos

    def replay_count(self, setnum = None):
        for remote in self.remotes:
            remote.send(('replay_count', setnum))
        results = [remote.recv() for remote in self.remotes]
        if setnum is not None:
            assert (np.array(results) == results[0]).all()
        return results[0]

    def stagger(self, maximum_length, prev_results):
        """stagger environments action steps, make every environment in different
            stage.
        """
        total = len(self.remotes)
        default_actions = self.get_default_action()
        """
        deltas = [0] * len(self.remotes)
        for i in range(len(self.remotes)):
            deltas[i] = maximum_length * i // total
        for step in range(max(deltas)):
            step_idx = []
            for idx in range(len(self.remotes)):
                if step < deltas[idx]:
                    step_idx.append(idx)
            for idx in step_idx:
                self.remotes[idx].send(('step', default_actions[idx]))
            for idx in step_idx:
                prev_results[idx] = self.remotes[idx].recv
        """
        for num, (default_action, remote) in enumerate(zip(default_actions, 
                                                           self.remotes)):
            delta = maximum_length * num // total
            for _ in range(delta):
                remote.send(('step', default_action))
                prev_results[num] = remote.recv()
        return prev_results
    
    def get_default_action(self):
        for remote in self.remotes:
            remote.send(('get_default_action', None))
        return [remote.recv() for remote in self.remotes]


def getCityFlowEnvVec(number, logpath, workpath, config, seed = 0, log = '', 
                      **kwargs):
    if not isinstance(config, list):
        config = [config] * number
    env_args = []
    for i in range(number):
        env_args.append([logpath + '/%s/' % i, 
                        workpath, 
                        config[i], 
                        log, 
                        seed + i, 
                        False])
        os.makedirs(env_args[-1][0], exist_ok = True)

    return EnvVecs(CityFlowEnv, 
                   number, 
                   env_args, 
                   is_args_list = True,
                   **kwargs)
