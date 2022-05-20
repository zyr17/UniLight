from .MultiAgentBase import *
from .MAL import MAL

"""Multi-Agent Learning with one Shared model, inherit from MAL for some
   function are common. 
"""


class NewModelNoCommuMultiAgent(MAL):
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
