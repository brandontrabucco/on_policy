from gym.spaces import Box
from gym import Env
import numpy as np


class PointmassEnv(Env):

    def __init__(self,
                 size=2,
                 order=2,
                 action_scale=0.1):
        self.observation_space = Box(-1.0 * np.ones([size * 2]), np.ones([size * 2]))
        self.action_space = Box(-1.0 * np.ones([size]), np.ones([size]))
        self.size = size
        self.order = order
        self.action_scale = action_scale
        self.position = np.zeros([self.size])
        self.goal = np.random.uniform(low=-1.0, high=1.0, size=[self.size])

    def reset(self,
              **kwargs):
        self.position = np.zeros([self.size])
        self.goal = np.random.uniform(low=-1.0, high=1.0, size=[self.size])
        return np.concatenate([self.position, self.goal], 0)

    def step(self,
             action):
        clipped_action = np.clip(action, -1.0 * np.ones([self.size]), np.ones([self.size]))
        scaled_action = clipped_action * self.action_scale
        self.position = np.clip(self.position + scaled_action,
                                np.ones([self.size]) * -1.0,
                                np.ones([self.size]))
        reward = -1.0 * np.linalg.norm(self.position - self.goal, ord=self.order)
        return np.concatenate([self.position, self.goal], 0), reward, False, {}

    def render(self, mode='human'):
        pass
