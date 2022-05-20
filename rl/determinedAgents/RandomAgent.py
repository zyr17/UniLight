from .DeterminedAgentBase import *


class RandomAgent(DeterminedAgentBase):
    def __init__(self, action, seed, **kwargs):
        self.phase_num = action[0]
        self.random = np.random.RandomState(seed)

    def get_action(self, state):
        return [[self.random.randint(self.phase_num)] for _ in state['TSflow']]
