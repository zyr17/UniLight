from .InnerModelBase import *


class LaneEmbedding(nn.Module):
    def __init__(self, observation, lane_embedding_size):
        super().__init__()
        self.observation = observation
        self.layers = []
        if isinstance(lane_embedding_size, int):
            lane_embedding_size = [lane_embedding_size]
        self.lane_embedding_size = lane_embedding_size
        self.input = 2
        last_input = self.input
        for hidden in lane_embedding_size:
            self.layers += [nn.Linear(last_input, hidden), nn.ReLU()]
            last_input = hidden
        self.layers = nn.Sequential(*self.layers)

    def forward(self, states):
        return self.layers(states)


class NewModelNoCommu(InnerModelBase):
    def __init__(self, dqn_hidden_size, lane_embedding_size, observation, 
                 lane_embedding_instance = None, **kwargs):
        super().__init__()
        self.hidden = dqn_hidden_size
        self.observation = observation
        self.phases = observation['TSphase']
        self.lane_number = observation['TSflow'][0]
        self.not_phases = []
        for phase in self.phases:
            one = []
            for num in range(self.lane_number):
                if num not in phase:
                    one.append(num)
            self.not_phases.append(one)
        self.weights = [5, 1]

        self.shared_lane = lane_embedding_instance is not None

        if lane_embedding_instance is not None:
            self.lane_embedding = lane_embedding_instance
        else:
            self.lane_embedding = LaneEmbedding(observation, 
                                                lane_embedding_size)

        self.fc = nn.Sequential(
            nn.Linear(lane_embedding_size * 2, dqn_hidden_size),
            nn.ReLU()
        )

    @staticmethod
    def default_wrapper(dueling_dqn, **kwargs):
        if dueling_dqn:
            return 'DuelingSplitModel'
        else:
            return 'DQNSplitModel'

    """if share lane embedding module, except it from named modules to avoid 
       multiple optimization
    """
    def named_modules(self, memo = None, prefix = ''):
        for n, p in super().named_modules(memo, prefix):
            if self.shared_lane:
                if n.find(prefix + '.lane_embedding') == 0:
                    continue
            yield n, p

    def replace_lane_embedding(self, lane_embedding):
        self.lane_embedding = lane_embedding
        self.shared_lane = True

    def forward(self, state):
        flow = cuda(torch.tensor(state['TSflow'])).float()
        green = cuda(torch.tensor(state['TSgreen'])).float()

        x = torch.stack((flow, green), dim = -1)
        x = self.lane_embedding(x)  # [BATCH, LANE, HIDDEN]

        res = []
        for phase, not_phase in zip(self.phases, self.not_phases):
            res.append(torch.cat((
                x[:, phase].mean(dim = 1) * self.weights[0],
                x[:, not_phase].mean(dim = 1) * self.weights[1]), dim = -1))
        res = torch.stack(res, dim = 1)  # [BATCH, PHASE, HIDDEN]
        res = self.fc(res)

        return res
