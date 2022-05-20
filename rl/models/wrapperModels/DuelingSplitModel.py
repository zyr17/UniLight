from .WrapperModelBase import *
from .DQNSplitModel import DQNSplitModel


class DuelingSplitModel(DQNSplitModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.V_obs = self.hidden * sum(self.actions)  # split embedding, so add
        self.V = nn.Linear(self.V_obs, 1)

    def forward(self, state):
        x = self.inner(state)
        V = self.V(x.reshape(*x.shape[:-2], -1))
        x = self.fc(x)
        x = x.squeeze(-1)
        mean = torch.mean(x, dim = -1).unsqueeze(-1)
        # print(x, mean, V)
        # print(x.shape, mean.shape, V.shape)
        return x + (V - mean).repeat(*([1] * (len(x.shape) - 1)), x.shape[-1])
