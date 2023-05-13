import gym
import numpy as np
from numpy import cos, pi, sin
from typing import Optional

from gym import spaces

class CustomReacher(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, init_bound, terminal_time):
        super(CustomReacher, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,  shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1e6, high=1e6, shape=(2,), dtype=np.float32)
        self.fr = np.array([0.025, 0.025]).reshape(2, 1)
        self.B = np.array([1, 1]).reshape(2, 1)
        self.gear = 0, 1
        self.dt = 0.01
        self.Q = np.diag(np.array([1, .1, 0, 0]))
        self.R = np.diag(np.array([1, 1]))
        self._iter = 0
        self.lb, self.ub = init_bound
        self._terminal_time = terminal_time
        self._env_id = env_id
        self.reward = 0


    def _terminated(self):
        return self._iter >= self._terminal_time
    def _get_reward(self, state, u):
        return -(state.T @ self.Q @ state+ u.T @ self.R @ u)

    def _enc_state(self):
        q1, q2, qd1, qd2 = self.state
        enc = lambda x: np.pi ** 2 * np.sin(x)
        return np.array([enc(q1), enc(q2), qd1, qd2], dtype=np.float32)

    def step(self, u):
        q1, q2, qd1, qd2 = self.state
        qd = np.array([qd1, qd2]).reshape(2, 1)
        a1, a2, a3 = 0.025 + 0.045 + 1.4 * 0.3 ** 2, 0.3 * 0.16, 0.045
        M11 = a1 + 2 * a2 * np.cos(q2)
        M12 = a3 + a2 * np.cos(q2)
        M22 = a3

        M = np.array([M11, M12, M12, M22]).reshape(2, 2)
        a4 = 0.3 * 0.1
        T1act = -qd2 * (2 * qd1 + qd2)
        T2act = qd1 ** 2
        Tbias = np.array([T1act, T2act]).reshape(2, 1)
        Tbias = -Tbias * (a4 * np.sin(q2))
        Tfric = self.fr * qd

        qdd = np.linalg.inv(M) @ (Tbias + u.reshape(2, 1)*self.B - Tfric).flatten()
        qdd1, qdd2 = qdd[0], qdd[1]

        qc_new = q1 + qd1 * self.dt
        qdc_new = qd1 + qdd1 * self.dt
        qp_new = q2 + qd2 * self.dt
        qdp_new = qd2 + qdd2 * self.dt
        self.state = np.array([qc_new, qp_new, qdc_new, qdp_new]).flatten()
        self.reward = self._get_reward(self._enc_state(), u)
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
        self.state = self.np_random.uniform(low=self.lb, high=self.ub, size=(4,))
        self._iter = 0
        if not return_info:
            return self.state
        else:
            return self.state, {}
