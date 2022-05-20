import numpy as np
import torch
import pdb
from collections import deque

from utils.utils import *


class RolloutStorages:
    """wrapper for several RolloutStorage."""
    def __init__(self, threads, *argv, **kwargs):
        self.rollouts = []
        self.threads = threads
        for _ in range(threads):
            self.rollouts.append(RolloutStorage(*argv, **kwargs))

    def call_rollout_functions(self, name):
        def func(*argv, **kwargs):
            res = []
            splitv = [[] for _ in self.rollouts]
            kwsplits = [{} for _ in self.rollouts]
            for arg in argv:
                alen = None
                try:
                    alen = len(arg)
                except Exception:
                    pass
                assert alen is None or alen == self.threads
                for num, i in enumerate(splitv):
                    if alen is None:
                        i.append(arg)
                    else:
                        i.append(arg[num])
            for key in kwargs:
                arg = kwargs[key]
                klen = None
                try:
                    klen = len(arg)
                except Exception:
                    pass
                assert klen is None or klen == self.threads
                for num, i in enumerate(kwsplits):
                    if alen is None:
                        i[key] = arg
                    else:
                        i[key] = arg[num]
            for rollout, av, kw in zip(self.rollouts, splitv, kwsplits):
                res.append(getattr(rollout, name)(*av, **kw))
            return res
        return func

    def cuda(self):
        self.call_rollout_functions('cuda')
        return self

    def collect_training_data(self, next_v):
        ret = self.call_rollout_functions('collect_training_data')(next_v)
        ret = [x for x in ret if x is not None]
        ret = zip(*ret)
        res = []
        for a in ret:
            if isinstance(a[0], list) or isinstance(a[0], tuple):
                res.append(sum(map(list, a), []))
            elif isinstance(a[0], np.ndarray):
                res.append(np.concatenate(a))
            else:  # tensor
                raise ValueError('error type' + str(a) + str(type(a)))
        if len(res) == 0:
            return None
        return res

    def __getattr__(self, name):
        return self.call_rollout_functions(name)


class RolloutStorage(object):
    """rollout storage for one environment. save multi-agent datas."""
    def __init__(self, num_steps, agent_number, obs_shape, action_space, gamma,
                 recurrent_hidden_state_size = None):

        assert recurrent_hidden_state_size is None  # not supported now
        self.agent_number = agent_number
        self.num_steps = num_steps
        self.GAMMA = gamma
        self.obs = []
        self.actions = np.zeros((num_steps, agent_number, 1), dtype = int)
        self.rewards = np.zeros((num_steps, agent_number, 1), dtype = float)
        self.preds = np.zeros((num_steps + 1, agent_number, 1), dtype = float)
        self.returns = np.zeros((num_steps + 1, agent_number, 1), 
                                dtype = float)
        self.probs = np.zeros((num_steps, agent_number, action_space[0][0]),
                                dtype = float)
        # ist[step] is 1, means obs[step + 1] is not following obs[step]
        self.ist = np.zeros((num_steps,), dtype = bool)
        # self.is_faket = torch.zeros_like(self.ist).bool()

        self.step = 0

    def cuda(self):
        """
        self.recurrent_hidden_states = self.recurrent_hidden_states.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        """
        # self.actions = self.actions.cuda()
        # self.rewards = self.rewards.cuda()
        # self.preds = self.preds.cuda()
        # self.ist = self.ist.cuda()
        # self.is_faket = self.is_faket.cuda()
        return self

    def full(self):
        return self.step == self.num_steps

    def reset(self, init_state):
        self.obs.clear()
        self.actions[:] = 0
        self.rewards[:] = 0
        self.ist[:] = 0
        # self.is_faket[:] = 0
        self.obs.append(init_state)
        self.step = 0

    def insert(self, obs, actions, rewards, ist, preds, probs, is_faket = None):
        #       recurrent_hidden_states, action_log_probs):
        # preds: value prediction 
        assert is_faket is None
        self.obs.append(obs)
        self.actions[self.step] = np.array(actions).copy()
        self.rewards[self.step] = np.array(rewards).copy()
        self.preds[self.step] = np.array(preds).copy()
        self.ist[self.step] = np.array(ist).copy()
        self.probs[self.step] = np.array(probs).copy()
        # self.is_faket[self.step + 1].copy_(torch.tensor(is_faket))

        self.step = (self.step + 1)  # % self.num_steps
        # pdb.set_trace()

    def collect_training_data(self, next_v):
        """return training data. If data is not fully collected, return None.
            else, return corresponding data with input agent and reset
        """
        if not self.full():
            return None
        self.compute_returns(next_v)
        ret_obs = self.obs[:-1]
        next_s = self.obs[1:]
        self.obs = self.obs[-1:]
        self.step = 0
        return (ret_obs, self.actions, self.returns[:-1], self.probs, next_s)

    def after_update(self):
        self.obs = [self.obs[-1]]
        # self.is_faket[0].copy_(self.is_faket[-1])
        self.step = 0

    def compute_returns(self,
                        next_value,
                        use_gae = False,
                        gae_lambda = 1,
                        stop_time_limit = True):
        self.returns[self.step] = next_value
        for step in reversed(range(self.step)):
            self.returns[step] = self.returns[step + 1] * \
                self.GAMMA * (1 - self.ist[step]) + self.rewards[step]


class CityFlowBuffer:
    def __init__(self, maxlen, observation, action):
        ''' use ndarray
        self.state = np.zeros((maxlen, threads, observation), dtype=float)
        self.action = np.zeros((maxlen, threads, action), dtype=float)
        self.reward = np.zeros((maxlen, threads, 1), dtype=float)
        self.next_s = np.zeros((maxlen, threads, observation), dtype=float)
        self.ist = np.zeros((maxlen, threads, 1), dtype=int)
        '''
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, data):
        self.data.append(data)

    def __setitem__(self, index, data):
        self.data[index] = data

    def __getitem__(self, index):
        if type(index) == np.ndarray:
            res = []
            for num in index:
                res.append(self.data[num])
            return res
        return self.data[index]


class ReplayBuffer:
    def __init__(self, n_steps, maxlen, batch_size, seed, buffer_instance):
        self.buffer = buffer_instance
        self.maxlen = maxlen
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.random = np.random.RandomState(seed)

        self.position = 0
        self.full = False
        self._reset = False
        self._state_deque = deque(maxlen = n_steps + 1)
        self._action_deque = deque(maxlen = n_steps)
        self._reward_deque = deque(maxlen = n_steps)
        self._ist_deque = deque(maxlen = n_steps)

    def __len__(self):
        return len(self.buffer)

    def cuda(self):
        return self

    def reset(self, state):
        self._reset = True
        self._state_deque.clear()
        self._action_deque.clear()
        self._reward_deque.clear()
        self._ist_deque.clear()
        self._state_deque.append(state)

    def append(self, action, reward, next_s, ist):
        assert(self._reset)
        self._state_deque.append(next_s)
        self._action_deque.append(action)
        self._reward_deque.append(reward)
        self._ist_deque.append(ist)
        if len(self._state_deque) == self.n_steps + 1:
            tot_reward = np.zeros((self.n_steps, *self._reward_deque[0].shape),
                                  dtype = float)
            tot_ist = np.zeros_like(self._ist_deque[0], dtype=int)
            for num, (reward, ist) in enumerate(zip(self._reward_deque, 
                                                    self._ist_deque)):
                tot_reward[num] = reward
                tot_ist += ist
            tot_reward = np.moveaxis(tot_reward, 0, -1)
            tot_ist -= self._ist_deque[-1]  # last state is terminal is ok
            data = [[], [], [], [], []]
            now_state_unpack = self._state_deque[0]
            next_state_unpack = self._state_deque[-1]
            for num, can_in in enumerate(tot_ist):
                if can_in == 0:
                    data[0].append(now_state_unpack[num])
                    data[1].append(self._action_deque[0][num])
                    data[2].append(tot_reward[num])
                    data[3].append(next_state_unpack[num])
                    data[4].append(self._ist_deque[-1][num])
            for one_data in zip(*data):
                self._buffer_append(*one_data)

    def _buffer_append(self, state, action, reward, next_s, ist):
        if len(self.buffer) < self.maxlen:
            self.buffer.append([state, action, reward, next_s, ist])
            if len(self.buffer) == self.maxlen:
                self.full = True
        else:
            self.buffer[self.position] = [state, action, reward, next_s, ist]
            self.position = (self.position + 1) % self.maxlen

    def sample(self):
        choice = self.random.choice(self.maxlen, 
                                    self.batch_size, 
                                    replace = False)
        return self.buffer[choice]
