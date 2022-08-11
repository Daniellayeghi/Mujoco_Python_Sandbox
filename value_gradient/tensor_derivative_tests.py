import numpy as np
from mujoco import derivative
from value_gradient.utilities.mj_utils import MjBatchOps
from value_gradient.utilities.mj_utils import MjBatchOps
from collections import namedtuple
import mujoco
import torch

if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(
        "/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml"
    )

    DataParams = namedtuple('DataParams', 'n_state, n_pos, n_vel, n_ctrl, n_desc, n_batch')
    d_params = DataParams(2, 1, 1, 1, 2, 32)
    d = mujoco.MjData(m)
    bo = MjBatchOps(m, d_params)
    d_vec = derivative.MjDataVecView(m, d)
    wrt = derivative.Wrt.State
    mode = derivative.Mode.Fwd
    params = derivative.MjDerivativeParams(1e-6, wrt, mode)
    dx = derivative.MjDerivative(m, d, params)
    res = dx.func()
    x = torch.rand(3, 32)
    bo.batch_df_fwd(x, dx)


