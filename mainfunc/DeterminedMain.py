from .MainFuncBase import *
from rl.multiAgent import IndependentDetermined
from rl.determinedAgents import *


class DeterminedMain(MainFuncBase):
    def __init__(self, env, agent, multi_agent, evaluate_round, render_steps, 
                 threads, seed, test_round,
                 # Specific args
                 log_folder, simulate_time, cityflow_log, 
                 **kwargs):

        work_folder = kwargs['work_folder']
        cityflow_config = kwargs['cityflow_config']
        self.dc = [0] * 3
        self.SEED = seed
        set_seed(seed)

        env = env.lower()
        if env == 'cityflow':
            self.env = getCityFlowEnvVec(threads, log_folder, work_folder, 
                                         cityflow_config, seed + 1, 
                                         cityflow_log)
        else:
            raise NotImplementedError('envs except cityflow is '
                                      'not implemented')
        if agent in globals():
            Agent = globals()[agent]
        else:
            raise NotImplementedError('unknown agent ' + agent)

        if multi_agent.lower() != 'independentdetermined':
            raise ValueError('DeterminedMain only supports '
                             'IndependentDetermined as multi-agent')

        MultiAgent = IndependentDetermined

        self.agent = MultiAgent(AgentClass = Agent,
                                observation = self.env.observation_space[
                                                  'intersections'], 
                                action = self.env.action_space,
                                env = self.env,
                                seed = self.SEED, 
                                **kwargs)

        self.RENDER = render_steps
        self._init_TXSW(**kwargs)

        assert threads == 1, 'only support one thread in determined!'

        self.THREADS = threads

        self.SIMULATE_TIME = simulate_time
        self.TOTAL_EPOCH = evaluate_round
        if test_round > 0:
            self.TOTAL_EPOCH = test_round

        self.PREVIOUS_REWARD = []
        self.PREVIOUS_DC = []
        self.PREVIOUS_RESULT = []
        self.epoch = 0

        log('DeterminedMain init over')

    def sampling(self, epoch):
        start_time = time.time()
        state, info = self.env.reset()
        action = self.agent.get_action(state)
        tot_reward = np.zeros(self.THREADS, dtype=float)
        for ite in range(self.SIMULATE_TIME):
            self.FRAME += 1
            next_s, reward, ist, infos = self.env.step(action)
            tot_reward += reward.sum(axis=1)
            next_action = self.agent.get_action(next_s)

            assert(ist.all() == ist.any())

            action = next_action
            state = next_s

        log(infos, level = 'TRACE')
        result = np.stack([x['average_time'] for x in infos]).mean()
        # dc_res = infos['dcroad_delay'].mean()
        self.PREVIOUS_REWARD.append(tot_reward)
        self.PREVIOUS_RESULT.append(result)
        # self.PREVIOUS_DC.append(dc_res)
        log('epoch %4d: reward=%.6f, result=%.6f' 
            % (epoch, tot_reward.mean(), result))
        self.TXSW.add_scalar('reward', tot_reward.mean(), self.FRAME)
        self.TXSW.add_scalar('time', result, self.FRAME)

    def main(self):
        self.FRAME = 0
        while self.epoch < self.TOTAL_EPOCH:
            self.sampling(self.epoch)
            self.epoch += 1
        final_result = np.array(self.PREVIOUS_RESULT)
        reward = np.array(self.PREVIOUS_REWARD)
        # dc_result = self.PREVIOUS_DC
        # dc_mean = sum(dc_result) / len(dc_result)
        # log('dc_mean and var:', dc_mean, max(dc_mean - min(dc_result), 
        #                                      max(dc_result) - dc_mean))
        log('reward:', reward.mean(), reward.std())
        log('result:', final_result.mean(), final_result.std())

    def __del__(self):
        self.TXSW.close()
