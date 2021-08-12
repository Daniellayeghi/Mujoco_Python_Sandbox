from collections import namedtuple
import numpy as np

CTRL_SIZE = 1
KLParams = namedtuple("KLParams", ["lambda", "samples", "time", "hes_reg"])
GradTraj = namedtuple("GradTraj",  ["mean", "variance"])
kl_params = KLParams(1, 30, 0, 1)
grad_traj = GradTraj([np.zeros(CTRL_SIZE) for _ in range(kl_params.time)],
                     [np.eye(CTRL_SIZE) for _ in range(kl_params.time)])


class KLControl(object):
    def __init__(self, traj, params):
        self._trajectory = traj.mean
        self._variance = traj.variance
        self._params = params

    def
