from .DeterminedAgentBase import *


class SOTLAgent(DeterminedAgentBase):
    def __init__(self, work_folder, cityflow_config, observation, action, 
                 sotl_green_threshold, sotl_red_threshold, 
                 sotl_min_switch, sotl_max_switch, **kwargs):
        self.observation_space = observation
        self.action_space = action
        self.phases = observation['TSphase']

        self.greent = sotl_green_threshold
        self.redt = sotl_red_threshold
        self.mins = sotl_min_switch
        self.maxs = sotl_max_switch

    def get_action(self, state):
        if len(state['TSphase']) > 1:
            log("Not support batch size > 1", level = 'ERROR')
            raise ValueError
        now_phase = int(state['TSphase'][0])
        time = state['TStime'][0]
        state = state['TSflow'][0]
        vnum = []
        if time < self.mins:
            return [[now_phase]]
        elif time > self.maxs:
            return [[(now_phase + 1) % len(self.phases)]]
        for p in self.phases:
            tot = 0
            for i in p:
                tot += state[i]
            if len(p) > 0:
                tot /= len(p)
            vnum.append(tot)

        next_phase = (now_phase + 1) % len(self.phases)
        if vnum[now_phase] <= self.greent and vnum[next_phase] > self.redt:
            return [[next_phase]]
        return [[now_phase]]
