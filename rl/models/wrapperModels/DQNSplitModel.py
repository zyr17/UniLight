from .WrapperModelBase import *
from .DQNBasicModel import DQNBasicModel

"""inner will return shape as [batch, TSphase, DCphase, hidden]. use fc to 
   [batch, TSphase, DCphase, 1] and resize to [batch, TSphase * DCphase]
"""


class DQNSplitModel(DQNBasicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fc = nn.Linear(self.hidden, 1)

    def forward(self, state):
        x = self.inner(state)
        x = self.fc(x)
        x = x.squeeze(-1)
        return x  # .reshape(*x.shape[:-2], -1)
