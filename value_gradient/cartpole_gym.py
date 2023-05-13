import gym
import numpy as np
from numpy import cos, pi, sin
from typing import Optional

from gym import spaces

class CustomCartPole(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, init_bound, terminal_time):
        super(CustomCartPole, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.m_p, self.m_c, self.l = .1, 1, 0.3
        self.fr = np.array([0, .1]).reshape(2, 1)
        self.g, self.gear = -9.81, 1
        self.dt = 0.01
        self.Q = np.diag(np.array([25, 25, 0.5, .1]))
        self.R = 0.1
        self._iter = 0
        self.lb, self.ub = init_bound
        self._terminal_time = terminal_time
        self._env_id = env_id


    def _terminated(self):
        return self._iter >= self._terminal_time
    def _get_reward(self, obs, u):
        return -(obs.T @ self.Q @ obs + self.R * u**2)

    def _enc_state(self):
        qc, qp, qdc, qdp = self.state
        enc = lambda x: np.pi ** 2 * np.sin(x)
        return np.array([qc, enc(qp), qdc, qdp], dtype=np.float32)

    def step(self, u):
        qc, qp, qdc, qdp = self.state
        qd = np.array([qdc, qdp]).reshape(2, 1)
        m_p, m_c, g, gear, l = self.m_p, self.m_c, self.g, self.gear, self.l
        M = np.array([m_p + m_c, m_p * l * np.cos(qp), m_p*l*np.cos(qp), m_p*l**2]).reshape(2, 2)
        C = np.array([0, -m_p*l*qdp*np.sin(qp), 0, 0]).reshape(2, 2)
        Tg = np.array([0, -m_p * g * l* np.sin(qp)]).reshape(2, 1)
        B = np.array([1, 0]).reshape(2, 1)

        qdd = (np.linalg.inv(M)@(-C@qd + Tg - self.fr*qd + B*u)).flatten()
        qddc, qddp = qdd[0], qdd[1]

        # qddc = 1/(m_c+m_p*np.sin(qp)**2)*(u+m_p*np.sin(qp)*(l*qdp**2+g*np.cos(qp)))
        # qddp = 1/(l*(m_c+m_p*np.sin(qp)**2))*(-u*np.cos(qp)-m_p*l*qdp**2*np.cos(qp)*np.sin(qp)-(m_c+m_p)*g*np.sin(qp) + qdp * fr_p)

        qc_new = qc + qdc * self.dt
        qdc_new = qdc + qddc * self.dt
        qp_new = qp + qdp * self.dt
        qdp_new = qdp + qddp * self.dt
        self.state = np.array([qc_new, qp_new, qdc_new, qdp_new]).flatten()
        reward = self._get_reward(self._enc_state(), u)[0]
        self._iter += 1

        terminate = self._terminated()

        return self.state, reward, terminate, False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        self.state = self.np_random.uniform(low=self.lb, high=self.ub, size=(4,)) * 0
        if not return_info:
            return self.state
        else:
            return self.state, {}
