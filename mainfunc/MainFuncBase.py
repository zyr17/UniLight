import torch
import numpy as np
import time
import pdb
import os
from utils.utils import *
from utils.log import log

from envs.SubprocVecEnv import getCityFlowEnvVec

import tensorboardX
TXSW = tensorboardX.SummaryWriter


class MainFuncBase:

    def _init_TXSW(self, enable_wandb, tensorboardx_comment, **kwargs):
        if enable_wandb:
            if tensorboardx_comment == '':
                tensorboardx_comment = None
            self.TXSW = WanDB_TXSW(tensorboardx_comment = tensorboardx_comment,
                                   **kwargs)
        elif tensorboardx_comment == '':
            self.TXSW = Fake_TXSW()
        else:
            self.TXSW = TXSW(comment = '_' + tensorboardx_comment)
