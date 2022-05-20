from .MultiAgentBase import *
from .MAL import MAL
from .MALShare import MALShare


class AllZero(torch.nn.Module):
    def __init__(self, output_number, *argv, **kwargs):
        super().__init__()
        self.output_number = output_number

    def forward(self, ids, *argv, **kwargs):
        return torch.zeros((len(ids), 
                            self.output_number)).float().to(ids.device)


class NewModelMultiAgent(MALShare):
    def __init__(self, AgentClass, observation, action, 
                 virtual_intersection_out_lines, **kwargs):
        self.direction_names = [
            'turn_left',
            'go_straight',
            'turn_right'
        ]
        self.N_STEPS = kwargs['n_steps']
        self.init_road_relation(observation)
        f_obs = [observation[0]] * len(observation)
        f_act = [action[0]] * len(action)
        for obs in observation:
            phase = obs['TSphase']
            phaseid2lanes = torch.zeros((len(phase), obs['TSflow'][0]))
            for num, p in enumerate(phase):
                for i in p:
                    phaseid2lanes[num][i] = 1
            phaseid2lanes = cuda(phaseid2lanes.float())
            obs['phaseid2lanes'] = phaseid2lanes

        self.out_road_embeddings = []
        v_out_lines = virtual_intersection_out_lines
        for _ in v_out_lines:
            self.out_road_embeddings.append(AllZero(
                output_number = len(self.direction_names)
            ))
        super().__init__(AgentClass, 
                         f_obs, 
                         f_act, 
                         road_relation = self.road_relation, 
                         out_road_embeddings = self.out_road_embeddings, 
                         **kwargs)
        self.out_road_embeddings = \
            torch.nn.ModuleList(self.out_road_embeddings)

    def named_modules(self, memo = None, prefix = ''):
        names = super().named_modules(memo=memo, prefix=prefix)
        for i in names:
            yield i
        emb = self.out_road_embeddings.named_modules(
            memo=memo, prefix=prefix + '.out_road_embeddings')
        for i in emb:
            yield emb

    def init_road_relation(self, observation):
        self.road_relation = []
        for obs in observation:
            assert self.direction_names == obs['DirectionNames']
            roadto = obs['RoadsOut']
            roadin = obs['RoadsIn']
            rloutbelong = obs['RoadLinksOut']
            rlinbelong = obs['RoadLinksIn']
            lanecount = obs['LaneCount']
            roadoutlanes = obs['RoadOutLanes']
            lanedirection = [self.direction_names.index(x) 
                             for x in obs['RoadLinkDirection']]
            ro2rl = [[-1] * 3 for _ in range(4)]
            for num, [i, j] in enumerate(zip(rloutbelong, lanedirection)):
                if i != -1:
                    ro2rl[i][j] = num
            ri2rl = [[-1] * 3 for _ in range(4)]
            for num, [i, j] in enumerate(zip(rlinbelong, lanedirection)):
                if i != -1:
                    ri2rl[i][j] = num
            # print(roadto, rloutbelong, road2rl)
            self.road_relation.append({
                'RoadsOut': roadto, 
                'RoadsIn': roadin, 
                'RoadLinksOut': rloutbelong, 
                'RoadLinksIn': rlinbelong, 
                'RoadOut2RoadLink': ro2rl, 
                'RoadIn2RoadLink': cuda(torch.tensor(ri2rl)), 
                'LaneCount': cuda(torch.tensor(lanecount)), 
                'RoadOutLanes': cuda(torch.tensor(roadoutlanes)).float(),
                'RoadLinkDirection': cuda(torch.tensor(lanedirection)),
            })
            # print(obs)
            # print(self.road_relation[-1])
            # pdb.set_trace()

    """extract data from next_s to state, then can cal loss of vehicle volume
    """
    def get_action_in_update_policy(self, samples_raw):
        # samples_raw: [BATCH, SAMPLETYPE=5, AGENT, ...data]
        samples = [x[:-1] + [[x[-1]] * len(self.agents)] for x in samples_raw]
        samples = list(zip(*samples))  # [S_TYPE, BATCH, AGENT, samples...data]
        state, action, reward, next_s, ist = [list(zip(*x)) for x in samples]
        # every data: [AGENT, BATCH, ...data]
        state = list(state)
        next_s = list(next_s)
        for num in range(len(self.agents)):
            # flatten inside every agent
            state[num] = flatten_data('dict', state[num])
            next_s[num] = flatten_data('dict', next_s[num])
            state[num]['NextInLaneVehicleNumber'] = \
                next_s[num]['InLaneVehicleNumber'][:, -self.N_STEPS]
            state[num]['NextPhase'] = \
                next_s[num]['TSprevphases'][:, -self.N_STEPS]
        # pdb.set_trace()
        state = self.forward(state, update_policy = 'state')
        next_s = self.forward(next_s, update_policy = 'next_s')
        # state: [AGENT, array(BATCH, ACTION)]

        samples = list(zip(state, action, reward, next_s, ist))
        # samples: [AGENT, S_TYPE, BATCH, ...data]
        return samples

