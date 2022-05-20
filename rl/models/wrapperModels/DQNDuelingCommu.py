from .WrapperModelBase import *
from .DQNDuelingModel import DQNDuelingModel


class DQNDuelingCommu(DQNDuelingModel):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        self.fc = nn.Linear(self.hidden * 2, self.action)
        self.V = nn.Linear(self.hidden * 2, 1)

    def forward(self, x):
        a = self.fc(x)
        mean = torch.mean(a, dim=-1).unsqueeze(-1)
        v = self.V(x)
        # print(a.shape, mean.shape, v.shape)
        return a + (v - mean).repeat(*([1] * (len(a.shape) - 1)), a.shape[-1])
