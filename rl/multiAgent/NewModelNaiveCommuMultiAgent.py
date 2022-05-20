from .MultiAgentBase import *
from .MAL import MAL
from .MALShare import MALShare

"""Multi-Agent Learning with one Shared model, inherit from MAL for some
   function are common. 
"""


class NewModelMultiAgent_MAL(MAL):
    def __init__(self, AgentClass, observation, action, **kwargs):
        self.share_embedding = True
        init_params = {
            'AgentClass': AgentClass, 
            'observation': observation,
            'action': action
        }

        # if share embedding, clear obs and action to cancel normal init, 
        # otherwise, do normal init
        if self.share_embedding:
            init_params['observation'] = []
            init_params['action'] = []
        super().__init__(**init_params, **kwargs)

        if self.share_embedding:
            # share embedding, rewrite init
            self.observation_space = observation
            self.action_space = action
            assert len(observation) == len(action)
            self.agents = torch.nn.ModuleList()
            self.embedding_instance = None
            for num, [obs, act] in enumerate(zip(observation, action)):
                kwargs['observation'] = obs
                kwargs['action'] = act
                kwargs['seed'] = self.random.randint(2 ** 31)
                kwargs['indices'] = [num]
                agent = AgentClass(**kwargs)
                self.agents.append(agent)
                if num == 0:
                    self.embedding_instance = agent.embedding_instance()
                else:
                    agent.embedding_instance(self.embedding_instance)
            for agent in self.agents:
                agent.init_communicate(
                    agents = self.agents,
                    all_observation = self.observation_space,
                    all_action = self.action_space,
                    **kwargs
                )


class NewModelMultiAgent_MALShare(MALShare):
    def __init__(self, AgentClass, observation, action, **kwargs):
        self.direction_names = [
            'turn_left',
            'go_straight',
            'turn_right'
        ]
        self.init_road_relation(observation)
        f_obs = [observation[0]] * len(observation)
        f_act = [action[0]] * len(action)
        super().__init__(AgentClass, f_obs, f_act, 
                         road_relation = self.road_relation, **kwargs)

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


NewModelNaiveCommuMultiAgent = NewModelMultiAgent_MALShare
