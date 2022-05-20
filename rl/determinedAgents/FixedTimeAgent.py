from .DeterminedAgentBase import *


class FixedTimeAgent(DeterminedAgentBase):
    def __init__(self, fixed_time, cityflow_config, observation, **kwargs):
        min_action_time = cityflow_config['MIN_ACTION_TIME']
        self.fixed_time = fixed_time
        self.min_action_time = min_action_time
        self.round = int(fixed_time / min_action_time)
        self.count = 0
        self.available = len(observation['TSphase'])

    def get_action(self, state):
        self.count += 1
        now_phase = int(state['TSphase'])
        if self.count == self.round:
            self.count = 0
            return [[(now_phase + 1) % self.available]]
        return [[now_phase]]
