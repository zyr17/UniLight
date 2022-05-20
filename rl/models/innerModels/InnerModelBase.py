import numpy as np
import torch
import torch.nn as nn
import pdb

from utils.utils import cuda
from utils.log import log


class InnerModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    """When model wrapper is not defined, the wrapper model will choose by 
       return of this function.
    """
    @staticmethod
    def default_wrapper(dqn_split_model, dueling_dqn, 
                        **kwargs):
        if dqn_split_model:
            if dueling_dqn:
                return 'DuelingSplitModel'
            else:
                return 'DQNSplitMix'
        elif dueling_dqn:
            return 'DQNDuelingModel'
        return 'DQNBasicModel'
