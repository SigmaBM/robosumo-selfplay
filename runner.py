import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, models, nsteps, nagent, anneal_bound):
        self.env = env
        self.models = models
        self.nenv = nenv = self.env.num_envs
        self.nagent = nagent
        self.obs = np.zeros((nenv,) + (len(env.observation_space.spaces),) + env.observation_space.spaces[0].shape,
                            dtype=models[0].train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = [model.initial_state for model in self.models]
        self.dones = np.array([[False for _ in range(self.nagent)] for _ in range(self.nenv)])
        # hyperparameter from "Emergent Complexity via Multi-Agent Competition"
        self.anneal_bound = anneal_bound

    @abstractmethod
    def run(self, update):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences with 1 trainable agent and old-version agents
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, models, nsteps, nagent, gamma, lam, anneal_bound=500):
        super().__init__(env=env, models=models, nsteps=nsteps, nagent=nagent, anneal_bound=anneal_bound)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self, update):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in tqdm(range(self.nsteps)):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            all_actions = []
            for agt in range(self.nagent):
                actions, values, self.states[agt], neglogpacs = self.models[agt].step(self.obs[:, agt, :],
                                                                                      S=self.states[agt],
                                                                                      M=self.dones[:, agt])
                if agt == 0:
                    mb_obs.append(self.obs[:, 0, :].copy())
                    mb_actions.append(actions)
                    mb_values.append(values)
                    mb_neglogpacs.append(neglogpacs)
                    mb_dones.append(self.dones[:, 0])
                all_actions.append(actions)
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            all_actions = np.stack(all_actions, axis=1)
            self.obs[:], _, self.dones, infos = self.env.step(all_actions)
            """
            info:
                shaping_reward:
                    ctrl_reward: The l_2 penalty on the actions to prevent jittery/unnatural movements
                    move_to_opp_reward: Reward at each time step proportional to magnitude of the velocity component 
                        towards the opponent.
                    push_opp_reward: The agent got penalty at each time step proportional to exp{-d_opp} where d_opp was
                        the distance of the opponent from the center the ring
                main_reward:
                    lose_penalty: -2000
                    win_reward: 2000
                episode_info: (when itersdones is True, automatically reset the env)

            """
            # exploration curriculum: r_t = alpha * shaping_reward + (1 - alpha) * done * main_reward
            alpha = 0
            if update <= self.anneal_bound:
                alpha = np.linspace(1, 0, self.anneal_bound)[update - 1]
            rewards = np.zeros(self.nenv)
            for e in range(self.nenv):
                rewards[e] = alpha * infos[e][0]['shaping_reward'] + (1 - alpha) * infos[e][0]['main_reward']
                maybeepinfo = infos[e][0].get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.models[0].value(self.obs[:, 0, :], S=self.states[0], M=self.dones[:, 0])

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones[:, 0]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_rewards)),
                mb_states[0], epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def f0(arr):
    """
    flatten axis 0 and 1
    """
    s = arr.shape
    return arr.reshape(s[0] * s[1], *s[2:])
