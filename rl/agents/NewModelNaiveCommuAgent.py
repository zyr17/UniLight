from .AgentBase import *
from .DQNAgent import DQNAgent


class NewModelNaiveCommuAgent(DQNAgent):
    def __init__(self, NM_lane_embedding_size, road_relation, **kwargs):
        self.hidden_size = kwargs['dqn_hidden_size']
        self.lane_embedding_size = NM_lane_embedding_size
        self.road_relation = road_relation
        super().__init__(lane_embedding_size = self.lane_embedding_size,
                         **kwargs)

    def embedding_instance(self, instance = None):
        if instance is not None:
            self.model_update.inner.replace_lane_embedding(instance[0])
            self.model_old.inner.replace_lane_embedding(instance[1])
            self.BEST_MODEL.inner.replace_lane_embedding(instance[2])
            self.opt = torch.optim.Adam(self.parameters(), self.LR)
        return [self.model_update.inner.lane_embedding,
                self.model_old.inner.lane_embedding,
                self.BEST_MODEL.inner.lane_embedding]

    def forward(self):
        if isinstance(self.selected_model, torch.nn.ModuleList):
            return [self.state2tensor, self.roadcomm_d, self.f_d]
        return [self.state2tensor, self.roadcomm_s, self.f_s]

    def state2tensor(self, states):
        res = []
        for state in states:
            res.append(self.model_update.inner.state2tensor(state))
        return res

    def arrange_rcomm(self, states, rcomm, state_key = 'rcomm'):
        assert len(states) == len(rcomm)
        default_rcomm = cuda(torch.zeros_like(rcomm[0][:, 0]))
        if state_key in states[0]:
            return states, rcomm
        for i in range(len(states)):
            onerc = []
            rlin = self.road_relation[i]['RoadLinksIn']
            rlfrom = self.road_relation[i]['RoadsIn'].copy()
            rlfrom.append([-1, -1])
            RLIN = [rlfrom[x] for x in rlin]

            # print(rlfrom, rlin)
            for ininter, inroad in RLIN:
                if ininter > -1 and inroad > -1:
                    onerc.append(rcomm[ininter][:, inroad])
                else:
                    onerc.append(default_rcomm)
            onerc = torch.stack(onerc, dim = 1)
            states[i][state_key] = onerc
        return states, rcomm

    def roadcomm_s(self, states):
        res = []
        for i in self.indices:
            res.append([
                states[i],
                self.selected_model.inner.roadcomm(states[i], 
                                                   self.road_relation[i])
            ])
        return res

    def f_s(self, states):
        states, rcomm = zip(*states)
        states, rcomm = self.arrange_rcomm(states, rcomm)
        res = []
        for i in self.indices:
            res.append(self.selected_model.forward(states[i]))
        return res

    def roadcomm_d(self, states):
        res = []
        for i in self.indices:
            res.append([
                states[i],
                self.selected_model[0].inner.roadcomm(states[i], 
                                                      self.road_relation[i]),
                self.selected_model[1].inner.roadcomm(states[i], 
                                                      self.road_relation[i]),
            ])
        return res

    def f_d(self, states):
        res = []
        states, rcomm_0, rcomm_1 = zip(*states)
        states, rcomm_0 = self.arrange_rcomm(states, rcomm_0, 'rcomm_0')
        for i in self.indices:
            states[i]['rcomm'] = states[i]['rcomm_0']
            res.append([self.selected_model[0].forward(states[i])])
        states, rcomm_1 = self.arrange_rcomm(states, rcomm_1, 'rcomm_1')
        for num, i in enumerate(self.indices):
            states[i]['rcomm'] = states[i]['rcomm_1']
            res[num].append(self.selected_model[1].forward(states[i]))
            res[num] = torch.stack(res[num], dim = 1)
        return res
