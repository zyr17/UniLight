from .MultiAgentBase import *

"""Multi-Agent Learning
"""


class MAL(MultiAgentBase):
    """When only want to call MultiAgentBase.__init__, pass observation and
       action as two empty lists
    """
    def __init__(self, AgentClass, observation, action, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation
        self.action_space = action
        assert len(observation) == len(action)
        self.agents = torch.nn.ModuleList()
        for num, [obs, act] in enumerate(zip(observation, action)):
            kwargs['observation'] = obs
            kwargs['action'] = act
            kwargs['seed'] = self.random.randint(2 ** 31)
            kwargs['indices'] = [num]
            self.agents.append(AgentClass(**kwargs))
        for agent in self.agents:
            agent.init_communicate(
                agents = self.agents,
                all_observation = self.observation_space,
                all_action = self.action_space,
                **kwargs
            )

    def cuda(self):
        for agent in self.agents:
            cuda(agent)
        return self

    def forward(self, state, **kwargs):
        for agent in self.agents:
            agent.forward_model_select(**kwargs)
        data = state
        funcs = [agent.forward() for agent in self.agents]
        funcs = zip(*funcs)
        for func_step in funcs:
            next_data = []
            for f in func_step:
                next_data += f(data)
            data = next_data
        return data

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
        state = self.forward(state, update_policy = 'state')
        next_s = self.forward(next_s, update_policy = 'next_s')
        # state: [AGENT, array(BATCH, ACTION)]

        samples = list(zip(state, action, reward, next_s, ist))
        # samples: [AGENT, S_TYPE, BATCH, ...data]
        return samples

    def _loss_backward(self, losses):
        final_loss = losses.sum()
        final_loss.backward()

    def update_policy(self, samples_raw, frame):
        samples = self.get_action_in_update_policy(samples_raw)
        losses = []
        for num, [agent, sample] in enumerate(zip(self.agents, samples)):
            losses.append(agent.calculate_loss(sample, frame, 
                                               'loss_%04d' % num))
        losses = torch.stack(losses)
        self._loss_backward(losses)
        for agent in self.agents:
            agent.optimizer_step()

    def unpack_states_in_action(self, states_raw):
        # pdb.set_trace()
        states = states_raw
        states = list(zip(*states))
        # output: [Agents, { every value: [Batch, (...data shape)] }]
        return [flatten_data('dict', x) for x in states]

    def action(self, states, eps = 0, model_name = 'update', **kwargs):
        # states: [Batch, Intersection, {data dict}]
        # return: [Batch, InterAgent, ActionsInOneInter]
        res = []
        states = self.unpack_states_in_action(states)
        with torch.no_grad():
            state_action = self.forward(states, model_name = model_name, 
                                        **kwargs) 
        for agent, s_a in zip(self.agents, state_action):
            res.append(agent.get_action(s_a, eps))
        # now res shape: [Agents, Batch, ActionNumForOneInter=1]
        res = list(zip(*res))
        return res

    def update_best(self):
        for agent in self.agents:
            agent.update_best()

    def log_model_structure(self):
        log('IQL, every model use same model structure and have different '
            'weight.', level = 'ALL')
        self.agents[0].log_model_structure()
