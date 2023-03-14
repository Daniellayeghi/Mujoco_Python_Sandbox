import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from collections import namedtuple
from time import sleep

State = namedtuple('State', 'time qpos qvel act udd_state')


class MjRenderer:

    def __init__(self, xml: str, dt=0.01):
        self.model = load_model_from_path(xml)
        self.data = MjSim(self.model)
        self._vel = np.zeros_like(self.data.data.qpos)
        self._ctrl = np.zeros_like(self.data.data.ctrl)
        self._set_state = lambda pos_t: State(time=0, qpos=pos_t, qvel=self._vel, act=self._ctrl, udd_state={})

        self.viewer = MjViewer(self.data)
        self.dt = dt

    def render(self, pos: np.array):
        for t in range(pos.shape[0]):
            state = self._set_state((pos[t]))
            mujoco_py.functions.mj_forward(self.model, self.data.data)
            self.data.set_state(state)
            self.viewer.render()
            sleep(self.dt)
