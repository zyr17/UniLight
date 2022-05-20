from .MultiAgentBase import *

"""For traditional algorithms
"""


class IndependentDetermined(MultiAgentBase):
    def __init__(self, AgentClass, observation, action, **kwargs):
        self.observation_space = observation
        self.action_space = action
        assert len(observation) == len(action)
        self.agents = []
        self.random = np.random.RandomState(kwargs['seed'])
        for num, [obs, act] in enumerate(zip(observation, action)):
            kwargs['observation'] = obs
            kwargs['action'] = act
            kwargs['seed'] = self.random.randint(2 ** 31)
            kwargs['index'] = num
            self.agents.append(AgentClass(**kwargs))

    def unpack_states_in_action(self, states):
        states = list(zip(*states))
        # output: [Agents, { every value: [Batch, (...data shape)] }]
        return [flatten_data('dict', x) for x in states]

    def get_action(self, states):
        # states: [Batch, Intersection, {data dict}]
        # return: [Batch, InterAgent, ActionsInOneInter]
        res = []
        states = self.unpack_states_in_action(states)
        # pdb.set_trace()
        for agent, state in zip(self.agents, states):
            res.append(agent.get_action(state))
        # res shape: [Agents, Batch, ActionNumForOneInter=1]
        res = list(zip(*res))
        return res

    def cuda(self):
        raise NotImplementedError('function not supported')

    def state_dict(self, model_file):
        raise NotImplementedError('function not supported')

    def update_policy(self, samples, frame):
        raise NotImplementedError('function not supported')

    def evaluate_action(self, states, model_name):
        raise NotImplementedError('function not supported')

    def update_best(self):
        raise NotImplementedError('function not supported')

    def save_model(self):
        raise NotImplementedError('function not supported')
