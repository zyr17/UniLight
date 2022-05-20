from .WrapperModelBase import *

"""Only for normal traffic control
"""


class DQNBasicModel(WrapperModelBase):
    def __init__(self, innerModel, **kwargs):
        super().__init__()

        self.actions = kwargs['action']
        # assert(len(self.action) == 1)
        self.action = 1
        for i in self.actions:
            self.action *= i

        self.random = np.random.RandomState(kwargs['seed'])
        kwargs['seed'] = self.random.randint(2 ** 31)

        self.hidden = kwargs['dqn_hidden_size']
        self.TXSW = None
        if 'TXSW' in kwargs:
            self.TXSW = kwargs['TXSW']

        self.inner = innerModel(**kwargs)

        # self.fc = nn.Linear(self.hidden, self.action.sum())
        self.fc = nn.Linear(self.hidden, self.action)

    def forward(self, state):
        x = self.inner(state)
        x = self.fc(x)
        return x
