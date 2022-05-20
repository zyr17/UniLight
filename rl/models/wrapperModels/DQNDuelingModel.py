from .WrapperModelBase import *
from .DQNBasicModel import DQNBasicModel


class DQNDuelingModel(DQNBasicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.V = nn.Linear(self.hidden, 1)

    def forward(self, state):
        x = self.inner(state)
        a = self.fc(x)
        mean = torch.mean(a, dim=-1).unsqueeze(-1)
        v = self.V(x)
        # print(a.shape, mean.shape, v.shape)
        return a + (v - mean).repeat(*([1] * (len(a.shape) - 1)), a.shape[-1])
