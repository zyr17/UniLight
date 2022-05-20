from .InnerModelBase import *


"""lane2embedding, predict phase
"""


class PhasePredict(nn.Module):
    def __init__(self, observation, NM_road_predict_hidden, **kwargs):
        super().__init__()
        self.observation = observation
        self.hidden = NM_road_predict_hidden
        self.direction_type = 3

        self.direction_hidden = 2
        self.emb_direction = nn.Embedding(self.direction_type, 
                                          self.direction_hidden)

        self.emb_fc = nn.Linear(2 + self.direction_hidden, self.hidden)

        self.attention = nn.MultiheadAttention(self.hidden, 1)
        self.phase_predict = nn.Sequential(
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )

        self.vol_predict_fc = nn.Linear(self.hidden, self.direction_type)

    def forward(self, states, road_relation):
        lanewidths = road_relation['LaneCount']  # [L]
        flow = states['TSflow']
        # wait = states['TSwait']
        green = states['TSgreen']
        direction = road_relation['RoadLinkDirection']
        direction = self.emb_direction(direction)
        direction = direction.unsqueeze(0).repeat(flow.shape[0], 1, 1)
        x = torch.stack((flow, green), dim = -1)
        x = self.emb_fc(torch.cat((x, direction), dim = -1))  # [B, L, H]
        q = k = v = x.transpose(0, 1)  # [L, B, H]
        x, x_w = self.attention(q, k, v)
        x = x.transpose(0, 1)  # [B, L, H]
        phase_predict = self.phase_predict(x)  # [B, L, 1]

        return x, phase_predict


"""input embedding and phase(predict or real), output every road traffic volume
   of every direction
"""


class VolumePredict(nn.Module):
    def __init__(self, observation, NM_road_predict_hidden, 
                 NM_scale_by_lane_number, **kwargs):
        super().__init__()
        self.observation = observation
        self.hidden = NM_road_predict_hidden
        self.lane_scale = NM_scale_by_lane_number
        self.direction_type = 3

        self.vol_predict_fc = nn.Linear(self.hidden, self.direction_type)

    def forward(self, lane_embedding, phase, road_relation):
        """calculate predicts for roadout
        """
        road2rl = road_relation['RoadOut2RoadLink']
        roadwidths = road_relation['RoadOutLanes']  # [R, self.direction_type]
        lane_embedding = lane_embedding * phase  # [B, L, H]
        res = []
        for rls, rwidth in zip(road2rl, roadwidths):
            assert len(rwidth) == self.direction_type
            selected = lane_embedding[:, rls]
            if self.lane_scale:
                selected = selected / (rwidth 
                                       + 1e-6).unsqueeze(0).unsqueeze(-1)
            res.append(selected.sum(dim = 1))
        res = torch.stack(res, dim = 1)  # [B, R, H]
        res = self.vol_predict_fc(res)  # [B, R, 3]
        return res


class LaneEmbedding(nn.Module):
    def __init__(self, observation, lane_embedding_size):
        super().__init__()
        self.observation = observation
        self.layers = []
        if isinstance(lane_embedding_size, int):
            lane_embedding_size = [lane_embedding_size]
        self.lane_embedding_size = lane_embedding_size
        self.input = 3  # flow, green, predict, predict_valid_mask
        last_input = self.input
        for hidden in lane_embedding_size:
            self.layers += [nn.Linear(last_input, hidden), nn.ReLU()]
            last_input = hidden
        self.layers = nn.Sequential(*self.layers)

    def forward(self, states):
        return self.layers(states)


class NewModel(InnerModelBase):
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

        self.phase_predict = PhasePredict(observation, **kwargs)
        self.volume_predict = VolumePredict(observation, **kwargs)

        if lane_embedding_instance is not None:
            self.lane_embedding = lane_embedding_instance
        else:
            self.lane_embedding = LaneEmbedding(observation, 
                                                lane_embedding_size)

        self.fc = nn.Sequential(
            nn.Linear(lane_embedding_size * 2 + 1,
                      dqn_hidden_size),
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

    @staticmethod
    def state2tensor(state):
        res = {}
        res['TSflow'] = cuda(torch.tensor(state['TSflow'])).float()
        res['TSgreen'] = cuda(torch.tensor(state['TSgreen'])).float()
        res['TSphase'] = cuda(torch.tensor(state['TSphase'])).long()
        res['Envtime'] = cuda(torch.tensor(state['Envtime'])).float()
        NILVN = 'NextInLaneVehicleNumber'
        if NILVN in state:
            res[NILVN] = cuda(torch.tensor(state[NILVN])).float()
        NP = 'NextPhase'
        if NP in state:
            res[NP] = cuda(torch.tensor(state[NP])).long()
        return res

    def forward(self, state):
        flow = state['TSflow']
        green = state['TSgreen']
        phase = state['TSphase']
        predict = state['predict']
        phase_s = nn.functional.one_hot(phase, num_classes = len(self.phases))
        phase_s = phase_s.float().unsqueeze(-1)  # [BATCH, PHASE, 1]

        x = torch.stack((flow, green, predict), dim = -1)
        x = self.lane_embedding(x)  # [BATCH, LANE, HIDDEN]

        res = []
        for i, [phase, not_phase] in enumerate(zip(self.phases, 
                                                   self.not_phases)):
            res.append(torch.cat((
                x[:, phase].mean(dim = 1) * self.weights[0],
                x[:, not_phase].mean(dim = 1) * self.weights[1], 
                phase_s[:, i]), dim = -1))
        res = torch.stack(res, dim = 1)  # [BATCH, PHASE, HIDDEN]
        res = self.fc(res)

        return res
