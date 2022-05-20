import torch
import numpy as np
import time
import pdb

from utils.utils import cuda, flatten_data
from utils.log import log

"""Base class of Agents.
    Args:
        seed: used to construct self.random
        index: a list of integers, means the agent indices controlled by this
            instance. when independent, the list contains only one integer; and
            if shared for all, it contains range(AGENT_NUM - 1). it is used for
            forward, when getting environment observations, it will calculate
            forward results of all listed indices and return a list of results.


    Functions:
        init_communicate: pass information used to communicate, such as other
            agents instance, agent connection graph, etc. default it won't do
            anything.
        state_dict: when model_file is set, load state dict from model file.
            return agent's state dict.
        cuda: call cuda(models)
        forward_model_select: select model for forwarding. `**kwargs` will
            contain informations from Main and multiAgent, such as 
            `evaluate=True`, `get=True` from main, `update_policy='state'` from
            multiAgent.update_policy. agent will choose proper model based on
            these information. specially, when is_train as bool is set, True
            will set model as train, False will set model as eval.
        forward: returns a list of functions, for example 
            `[self.f1, self.f2, self.f3]`. 
            the design is for the agent can't forward simply, which 
            is usually happen when it needs to use observations and features 
            of other agents. every function takes previous function's return 
            as input, except first function takes state as input. here input 
            contains all agents datas, and agent should pick out information it 
            needed. for example, with 2 agents a1 and a2, for `self.f1` they 
            all take [o1, o2] as input, and return `res_1_1` and `res_1_2`. 
            then `self.f2` takes `[res_1_1, res_1_2]` as input. if one agent 
            need other agent's information, the agent can get it simply. if 
            not, it can only take `state[self.index]` as input. so multiAgent 
            should guarentee that when calling `self.fx`, `self.f(x-1)` of all 
            other agents has called, so the agent can communicate to other 
            agents using data calculated in previous functions, which means 
            multiAgent may collect all agent's forward functions, then zip 
            them, and run them with input or last output. if communication is 
            not used, for independent agents, just `return [self.f_func]`, and 
            `self.f_func` contains one line 
            `return [self.selected_model.forward(state[self.index[0]])]`.
        update_best: save current model as the best model.
        calcudate_reward: calculate n-step reward to one final reward.
            input: np.array, last dim is n-step
        get_action: to get action with state_q and eps (used in DQN, can ignore
            it when using actor-critic).
        log_model_structure: print model structure to log
        calculate_loss: calculate and return loss by given samples, and if 
            `txsw_name` is not empty, write log to tensorboard with frame.
        optimizer_step: make optimizer step
"""


class AgentBase(torch.nn.Module):
    def __init__(self, seed, indices, *argv, **kwargs):
        super().__init__()
        self.random = np.random.RandomState(seed)
        self.indices = indices

    def init_communicate(self, *argv, **kwargs):
        pass

    def cuda(self):
        raise NotImplementedError()

    def forward_model_select(self, **kwargs):
        raise NotImplementedError()

    def forward(self, state, model_name):
        raise NotImplementedError()

    def update_best(self):
        raise NotImplementedError()

    def calculate_reward(self, reward):
        raise NotImplementedError()

    def get_action(self, state_q, eps):
        raise NotImplementedError()

    def log_model_structure(self):
        raise NotImplementedError()

    def calculate_loss(self, samples, frame, txsw_name = ''):
        raise NotImplementedError()

    def optimizer_step(self):
        raise NotImplementedError()
