import torch
import numpy as np
import time
import pdb

from utils.utils import cuda, flatten_data, get_intersection_info
from utils.utils import unpack_flattened_data
from utils.log import log

"""Base class of MultiAgent
"""


class MultiAgentBase(torch.nn.Module):
    def __init__(self, seed, **kwargs):
        super().__init__()
        self.random = np.random.RandomState(seed)

    def cuda(self):
        raise NotImplementedError()
    """backward loss of all models used by self.update_policy. if need to 
       backward through communication, it's hard to backward every agents 
       separately. 
    """

    def _loss_backward(self, losses):
        raise NotImplementedError()
    """samples is a list with N samples. each sample contains datas 
       [state, action, reward, next_s, terminate]. First four data are all 
       lists with number of intersections as length, and terminate is one bool.
    """

    def update_policy(self, samples, frame):
        raise NotImplementedError()

    def get_action(self, states, eps):
        raise NotImplementedError()

    def evaluate_action(self, states, model_name):
        raise NotImplementedError()

    def update_best(self):
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()
