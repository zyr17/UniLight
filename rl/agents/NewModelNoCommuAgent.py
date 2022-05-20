from .AgentBase import *
from .DQNAgent import DQNAgent


class NewModelNoCommuAgent(DQNAgent):
    def __init__(self, NM_lane_embedding_size, **kwargs):
        self.hidden_size = kwargs['dqn_hidden_size']
        self.lane_embedding_size = NM_lane_embedding_size
        super().__init__(lane_embedding_size = self.lane_embedding_size,
                         **kwargs)

    def embedding_instance(self, instance = None):
        if instance is not None:
            self.model_update.inner.replace_lane_embedding(instance[0])
            self.model_old.inner.replace_lane_embedding(instance[1])
            self.BEST_MODEL.inner.replace_lane_embedding(instance[2])
            self.opt = torch.optim.Adam(self.parameters(), self.LR)
        return [self.model_update.inner.lane_embedding,
                self.model_old.inner.lane_embedding,
                self.BEST_MODEL.inner.lane_embedding]
