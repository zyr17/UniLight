from .AgentBase import *

"""Base DQN agent.
    Args:
        model_old, model_update: model instance
        gamma, dqn_net_update_freq, learning_rate, n_steps: parameters
        double_dqn: True if use double dqn structure

    Functions:
        cuda: make no effect now
        opt_state_dict: set or get state dict of self.opt
        calculate_reward: input reward mat [*, N_STEP], multiply gamma and 
            return final result [*]
        calculate_loss: input samples, calculate and return loss with DQN 
            policy. frames and TXSW is used to write logs.
        get_action: get action from model and return it to main.
"""


class DQNAgent(AgentBase):
    def __init__(self, wrapperModel, 
                 observation, action, innerModel, TXSW,  # wrapperModel param
                 gamma, dqn_net_update_freq, learning_rate, double_dqn, 
                 seed, n_steps, **kwargs):
        super().__init__(seed, **kwargs)
        self.TXSW = TXSW
        self.observation = observation
        self.action = action
        model_args = {
            'observation': observation,
            'action': action,
            'innerModel': innerModel,
            'TXSW': TXSW
        }
        self.model_old = cuda(wrapperModel(
            **model_args, 
            seed = self.randint(), **kwargs))
        self.model_update = cuda(wrapperModel(
            **model_args, 
            seed = self.randint(), **kwargs))
        self.BEST_MODEL = cuda(wrapperModel(
            **model_args, 
            seed = self.randint(), **kwargs))
        self.model_old.eval()  # a.k.a target net
        self.model_update.train()
        self.BEST_MODEL.eval()
        self.GAMMA = gamma
        self.UPDATE = dqn_net_update_freq
        self.LR = learning_rate
        self.DOUBLE = double_dqn
        self.N_STEPS = n_steps

        self.update_count = 0
        self.opt = torch.optim.Adam(self.parameters(), self.LR)

        self.loss = torch.nn.MSELoss()

        # add state dict of optimizer to state_dict
        def add_opt_state_dict_hook(self, state_dict, prefix, local_metadata):
            opt_state_dict = self.opt.state_dict()
            state_dict[prefix + 'opt'] = opt_state_dict
            # for k, v in opt_state_dict.items():
            #     state_dict[prefix + k] = v
        self._register_state_dict_hook(add_opt_state_dict_hook)

        # remove selected_model data
        def remove_selected_hook(self, state_dict, prefix, local_metadata):
            select = prefix + 'selected_model'
            keys = list(state_dict.keys())
            for k in keys:
                if k.find(select) == 0:
                    del state_dict[k]
        self._register_state_dict_hook(remove_selected_hook)

    """read optimizer state dict and delete it from dict
    """
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        opt_key = prefix + 'opt'
        if opt_key in state_dict:
            self.opt.load_state_dict(state_dict[opt_key])
            del state_dict[opt_key]
        elif strict:
            missing_keys.append(opt_key)
        self.selected_model = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, 
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def named_modules(self, memo = None, prefix = ''):
        # only self.model_update will be trained
        return self.model_update.named_modules(memo, prefix + '.model_update')

    def randint(self, k = 2 ** 31):
        return self.random.randint(k)

    def cuda(self):
        self.model_update = cuda(self.model_update)
        self.model_old = cuda(self.model_old)
        self.BEST_MODEL = cuda(self.BEST_MODEL)
        return self

    def forward_model_select(self, **kwargs):
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
            if model_name.lower() == 'update':
                model = self.model_update
            elif model_name.lower() == 'old':
                model = self.model_old
            elif model_name.lower() == 'best':
                model = self.BEST_MODEL
        elif 'update_policy' in kwargs:
            datasource = kwargs['update_policy']
            self.model_update.train()
            self.model_old.eval()
            if datasource == 'state':
                model = self.model_update
            elif datasource == 'next_s':
                if self.DOUBLE:
                    model = torch.nn.ModuleList()
                    model.append(self.model_old)
                    model.append(self.model_update)
                else:
                    model = self.model_old
            else:
                raise ValueError("can't determine which model to select")
        else:
            raise ValueError("can't determine which model to select")

        if 'evaluate' in kwargs or 'get' in kwargs:
            model.eval()

        if 'is_train' in kwargs:
            if kwargs['is_train']:
                model.train()
            else:
                model.eval()

        # print(kwargs, 
        #       (model == self.model_update) * 'update', 
        #       (model == self.model_old) * 'old', 
        #       (model == self.BEST_MODEL) * 'best',
        #       isinstance(model, list) * 'update old'
        # )
        self.selected_model = model

    def forward(self):
        if isinstance(self.selected_model, torch.nn.ModuleList):
            return [self.forward_double]
        return [self.forward_single]

    """for double q learning, next_s will forward through both old and 
       update. both results of two will return as a tuple.
    """
    def forward_double(self, states):
        res = []
        for idx in self.indices:
            res.append(torch.stack([self.model_old.forward(states[idx]),
                                    self.model_update.forward(states[idx])],
                                   dim = 1))
        # res: [BATCH, 2, ACTION]
        return res

    def forward_single(self, states):
        res = []
        for idx in self.indices:
            res.append(self.selected_model.forward(states[idx]))
        return res

    def update_best(self):
        self.BEST_MODEL.load_state_dict(self.model_update.state_dict())

    def calculate_reward(self, reward):
        gamma = self.GAMMA ** np.arange(reward.shape[-1])
        reward = (reward * gamma).sum(-1)
        return reward

    def get_action(self, state_action, eps):
        rand_actions = self.random.randint(self.action, 
                                           size = state_action.shape[:-1])
        q = state_action
        q = torch.argmax(q, dim=-1).detach().cpu().numpy()  # [Batch]
        randidx = self.random.rand(*state_action.shape[:-1]) < eps
        q[randidx] = rand_actions[randidx]
        q = np.expand_dims(q, axis = -1)  # [Batch, 1]
        return q  # return: [Batch, ActionNumForOneInter=1]

    def log_model_structure(self):
        log('model structure:\n', self.model_old, '\n-------', level = 'ALL')

    def calculate_loss(self, samples, frame, txsw_name = ''):
        self.model_old.eval()
        self.model_update.train()
        # each of 5 is a list contains BATCHNUM items.
        state_q, action, reward, next_s_q, ist = samples
        if self.DOUBLE:
            # next_s_q: [BATCH, 2, ACTION]
            next_s_q_old = next_s_q[:, 0]
            next_s_q_update = next_s_q[:, 1]

        action = np.array(action)
        reward = np.array(reward)
        reward = self.calculate_reward(reward)
        ist = np.array(ist)
        self.opt.zero_grad()
        action = cuda(torch.tensor(action).long())
        reward = cuda(torch.tensor(reward).float())

        # combine one intersection's actions
        # action = action.mul(cuda(torch.tensor([1, 8]))).sum(-1)
        action.squeeze_(-1)  # when every intersetion has only one action
        ist = 1 - np.array(ist)

        if self.DOUBLE:
            next_a = next_s_q_update.max(dim = -1)[1]
            next_q = next_s_q_old.gather(1, next_a.unsqueeze(-1)).squeeze(1)
        else:
            next_q = next_s_q.max(dim = -1)[0]
        reward_b = (reward 
                    + (self.GAMMA ** self.N_STEPS) 
                    * next_q 
                    * cuda(torch.tensor(1 - ist).float()))
        q = state_q.gather(1, action.unsqueeze(-1)).squeeze(1)
        L = self.loss(q, reward_b)
        # L.backward()
        # self.opt.step()
        if txsw_name != '':
            self.TXSW.add_scalar(txsw_name, L.item(), frame)
        # return L.item()
        return L

    def optimizer_step(self):
        self.opt.step()
        self.opt.zero_grad()
        self.update_count += 1
        if self.update_count % self.UPDATE == 0:
            self.model_old.load_state_dict(self.model_update.state_dict())
