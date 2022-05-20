from .MultiAgentBase import *
from .MAL import MAL

"""Multi-Agent Learning with one Shared model, inherit from MAL for some
   function are common. 
"""


class MALShare(MAL):
    def __init__(self, AgentClass, observation, action, **kwargs):
        # remove observation and action, so MAL.__init__ won't create any agent
        super().__init__(AgentClass = AgentClass, 
                         observation = [],
                         action = [],
                         **kwargs)
        # new init, only generate one agent instance. 
        self.observation_space = observation
        self.action_space = action
        assert len(observation) == len(action)
        """
        for obs in observation:
            if observation[0] != obs and False:
                log('observations not all same! cannot use MALShare', 
                    level = 'ERROR')
                raise ValueError
        for act in action:
            if action[0] != act:
                log('actions not all same! cannot use MALShare', 
                    level = 'ERROR')
                raise ValueError
        """
        kwargs['observation'] = observation[0]
        kwargs['action'] = action[0]
        kwargs['seed'] = self.random.randint(2 ** 31)
        self.agent = AgentClass(indices = list(range(len(action))), **kwargs)
        # for inherit functions
        for _ in action:
            self.agents.append(self.agent)
        self.agent.init_communicate(
            agents = self.agents,
            all_observation = observation,
            all_action = action,
            **kwargs
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, 
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def named_modules(self, memo = None, prefix = ''):
        # self.agents is fake, only self.agent.parameters() is valid
        return self.agent.named_modules(memo, prefix + '.agent')

    def forward(self, state, **kwargs):
        for agent in self.agents:
            agent.forward_model_select(**kwargs)
        data = state
        funcs = self.agent.forward()
        for func_step in funcs:
            data = func_step(data)
        return data

    def _loss_backward(self, losses):
        losses.backward()

    def update_policy(self, samples_raw, frame):
        samples = self.get_action_in_update_policy(samples_raw)
        # gather all sample as one and feed
        sample = [[] for _ in samples[0]]
        for s in samples:
            for sample_one, s_one in zip(sample, s):
                sample_one.extend(s_one)
        sample[0] = torch.stack(sample[0])
        sample[3] = torch.stack(sample[3])
        # update only agent
        L = self.agent.calculate_loss(sample, frame, 'loss')
        self._loss_backward(L)
        self.agent.optimizer_step()

    def action(self, states, eps = 0, model_name = 'update', **kwargs):
        # states: [Batch, Intersection, {data dict}]
        # return: [Batch, InterAgent, ActionsInOneInter]
        states = self.unpack_states_in_action(states)
        with torch.no_grad():
            state_action = self.forward(states, model_name = model_name, 
                                        **kwargs) 
        res = self.agent.get_action(torch.stack(state_action), eps)
        # now res shape: [Agents, Batch, ActionNumForOneInter=1]
        res = list(zip(*res))
        return res

    def update_best(self):
        # only one model
        self.agent.update_best()

    def log_model_structure(self):
        log('IQLShare, every model use same model structure and have same '
            'weight.', level = 'ALL')  # different log
        self.agent.log_model_structure()
