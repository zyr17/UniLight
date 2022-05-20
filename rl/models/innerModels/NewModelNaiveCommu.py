from .InnerModelBase import *


"""deal for all 
"""


class RoadCommunication(nn.Module):
    def __init__(self, observation, NM_road_predict_hidden, 
                 NM_scale_by_lane_number, **kwargs):
        super().__init__()
        self.observation = observation
        self.hidden = NM_road_predict_hidden
        self.lane_scale = NM_scale_by_lane_number

        self.fc = nn.Linear(2, self.hidden)
        self.attention = nn.MultiheadAttention(self.hidden, 1)

    def forward(self, states, road_relation):
        road2rl = road_relation['RoadOut2RoadLink']
        roadwidths = road_relation['RoadOutLanes']  # [R]
        lanewidths = road_relation['LaneCount']  # [L]
        flow = states['TSflow']
        green = states['TSgreen']
        x = self.fc(torch.stack((flow, green), dim = -1))  # [B, L, H]
        if self.lane_scale:
            x = x * lanewidths.unsqueeze(-1).unsqueeze(0)
        q = x.mean(dim = -2, keepdim = True).transpose(0, 1)  # [1, B, H]
        res = []
        for rls, rwidth in zip(road2rl, roadwidths):
            """same as below, but may calculate quicker in CPU?
            k = []
            for num, rlid in enumerate(rls):
                if not rlid:
                    k.append(x[:, num])
                    print(k[-1].shape)
            k = v = torch.stack(k, dim = 0) # [X, B, H]
            att1, att_w1 = self.attention(q, k, v)
            """

            k = v = x.transpose(0, 1)[rls]  # [L, B, H]
            # att, att_w = self.attention(q, k, v)
            att = v.mean(0, keepdim = True)

            if self.lane_scale:
                srw = sum(rwidth)
                if srw == 0:
                    srw = 999
                att = att / srw
            res.append(att.squeeze(0))
            # if flow.shape[0] > 1: pdb.set_trace()
        res = torch.stack(res, dim = 1)  # [B, R, H]

        """not work on old version of pytorch, not tested on new version
        R = len(self.road2rl)
        k = v = x.transpose(0, 1).repeat_interleave(R, dim = 1) # [L, B*R, H]
        q = x.mean(dim = -2, keepdim = True).transpose(0, 1)
        q = q.repeat_interleave(R, dim = 1) # [1, B*R, H]
        mask = self.road2rl.repeat(flow.shape[0], 1)
        att, att_w = self.attention(q, k, v, attn_mask = mask)
        res = att.reshape(-1, R, self.hidden)
        if res.shape[0] > 1:
            pdb.set_trace()
        """

        return res


class LaneEmbedding(nn.Module):
    def __init__(self, observation, lane_embedding_size, rcomm_hidden):
        super().__init__()
        self.observation = observation
        self.layers = []
        if isinstance(lane_embedding_size, int):
            lane_embedding_size = [lane_embedding_size]
        self.lane_embedding_size = lane_embedding_size
        self.input = 2 + rcomm_hidden
        last_input = self.input
        for hidden in lane_embedding_size:
            self.layers += [nn.Linear(last_input, hidden), nn.ReLU()]
            last_input = hidden
        self.layers = nn.Sequential(*self.layers)

    def forward(self, states):
        return self.layers(states)


class NewModelNaiveCommu(InnerModelBase):
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

        self.roadcomm = RoadCommunication(observation, **kwargs)
        self.rcomm_hidden = self.roadcomm.hidden

        if lane_embedding_instance is not None:
            self.lane_embedding = lane_embedding_instance
        else:
            self.lane_embedding = LaneEmbedding(observation, 
                                                lane_embedding_size,
                                                self.rcomm_hidden)

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

    @staticmethod
    def state2tensor(state):
        res = {}
        res['TSflow'] = cuda(torch.tensor(state['TSflow'])).float()
        res['TSgreen'] = cuda(torch.tensor(state['TSgreen'])).float()
        return res

    def forward(self, state):
        flow = state['TSflow']
        green = state['TSgreen']
        rcomm = state['rcomm']

        x = torch.stack((flow, green), dim = -1)
        x = torch.cat((x, rcomm), dim = -1)
        x = self.lane_embedding(x)  # [BATCH, LANE, HIDDEN]

        res = []
        for phase, not_phase in zip(self.phases, self.not_phases):
            res.append(torch.cat((
                x[:, phase].mean(dim = 1) * self.weights[0],
                x[:, not_phase].mean(dim = 1) * self.weights[1]), dim = -1))
        res = torch.stack(res, dim = 1)  # [BATCH, PHASE, HIDDEN]
        res = self.fc(res)

        return res
