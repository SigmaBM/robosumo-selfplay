from gym.core import Wrapper
import time


class SumoEnv(Wrapper):
    def __init__(self, env, allow_early_resets=False, reset_keywords=(), info_keywords=(), file_prefix=""):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.dense_rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your"
                               " env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.dense_rewards = []
        self.needs_reset = False
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew[0])
        self.dense_rewards.append(info[0]['shaping_reward'])
        if done[0]:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epdenserew = sum(self.dense_rewards)
            epinfo = {"r": round(eprew, 6), "dr": round(epdenserew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[0][k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            info[0]['episode'] = epinfo
        self.total_steps += 1
        return ob, rew, done, info

    def close(self):
        self.env.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times
