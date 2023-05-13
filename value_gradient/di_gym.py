import gym
import numpy as np
from numpy import cos, pi, sin
from typing import Optional

from gym import spaces

class CustomDoubleIntegrator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, init_bound, terminal_time):
        super(CustomDoubleIntegrator, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1e6, high=1e6, dtype=np.float32)
        self.fr = .1
        self.M = 1
        self.gear = 1
        self.dt = 0.01
        self.Q = np.diag(np.array([1, .1]))
        self.R = .001
        self._iter = 0
        self.lb, self.ub = init_bound
        self._terminal_time = terminal_time
        self._env_id = env_id
        self.prev_reward = 0
        self._init_state = self.np_random.uniform(low=self.lb, high=self.ub, size=(2,))


    def _terminated(self):
        cond = (self._iter >= self._terminal_time)
        return cond
    def _get_reward(self, state, u):
        return -((state.T @ self.Q @ state) + self.R*u**2)

    def _enc_state(self):
        q, qd = self.state
        return np.array([q, qd], dtype=np.float32)

    def step(self, u):
        q, qd = self.state
        qdd = (1/self.M * (u*self.gear - qd * self.fr))[0]

        q_new = q + qd * self.dt
        qd_new = qd + qdd * self.dt
        self.reward = self._get_reward(self._enc_state(), u)[0]
        self.mean_reward = (self.reward  + self.prev_reward)/2
        self.prev_reward = self.prev_reward
        self.state = np.array([q_new, qd_new]).flatten()
        self._iter += 1

        terminate = self._terminated()

        return self.state, self.reward, terminate, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        self._iter = 0
        self.state = self._init_state
        self.prev_reward = 0
        self.mean_reward = 0
        if not return_info:
            return self.state
        else:
            return self.state, {}
