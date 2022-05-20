import numpy as np
import torch
import torch.nn as nn
import pdb

from utils.utils import cuda
from utils.log import log


class WrapperModelBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

