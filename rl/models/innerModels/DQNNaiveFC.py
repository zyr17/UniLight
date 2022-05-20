from .InnerModelBase import *


class DQNNaiveFC(InnerModelBase):
    def __init__(self, observation, dqn_hidden_size,
                 use_predict = False, **kwargs):
        super(DQNNaiveFC, self).__init__()

        self.use_predict = use_predict

        # cityflow specific
        self.observation = observation['TSflow'][0]
        self.phase_obs = len(observation['TSphase'])
        self.green_obs = observation['TSgreen'][0]

        self.phase_emb = torch.nn.Embedding(self.phase_obs, self.phase_obs)
        self.inputlen = self.observation + self.phase_obs + self.green_obs
        if use_predict:
            self.inputlen += self.observation

        self.fc1 = nn.Sequential(
            nn.Linear(self.inputlen, dqn_hidden_size * 8),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dqn_hidden_size * 8, dqn_hidden_size),
            nn.ReLU(),
        )

    def forward(self, state):
        flow = cuda(torch.tensor(state['TSflow'])).float()
        phase = cuda(torch.tensor(state['TSphase'])).long()
        green = cuda(torch.tensor(state['TSgreen'])).float()
        phase = self.phase_emb(phase)
        cat = [flow, phase, green]
        if self.use_predict:
            flow = cuda(torch.tensor(state['predict'])).float()
            cat.append(flow)
        x = self.fc1(torch.cat(cat, dim=-1))
        x = self.fc2(x)
        return x
