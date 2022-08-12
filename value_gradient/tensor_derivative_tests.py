import numpy as np
from mujoco import derivative
from value_gradient.utilities.mj_utils import MjBatchOps
from value_gradient.utilities.mj_utils import MjBatchOps
from collections import namedtuple
import mujoco
import torch
from mujoco.derivative import *
if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(
        "/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml"
    )

    batch_size = 2
    DataParams = namedtuple('DataParams', 'n_state, n_pos, n_vel, n_ctrl, n_desc, n_batch')
    d_params = DataParams(4, 2, 2, 1, 2, batch_size)
    d = mujoco.MjData(m)
    bo = MjBatchOps(m, d_params)
    x = torch.zeros(batch_size, d_params.n_state)
    full_x = torch.zeros(batch_size, d_params.n_state + d_params.n_vel)
    u = torch.zeros(batch_size, d_params.n_ctrl)
    res_dfdx = torch.zeros(batch_size, d_params.n_state * d_params.n_state)
    res_dfdu = torch.zeros(batch_size, d_params.n_state * d_params.n_ctrl)
    res_dfinvdx = torch.zeros(batch_size, (d_params.n_state + d_params.n_vel) * d_params.n_vel)
    res_dfdt = torch.zeros(batch_size, d_params.n_state)
    res_qfrc = torch.zeros(batch_size, d_params.n_vel)

    bo.b_dfdx(res_dfdx, x, u)
    bo.b_dfdu(res_dfdu, x, u)
    bo.b_dfinv_full_x(res_dfinvdx, full_x)
    bo.b_f_x(res_dfdt, x, u)
    bo.b_finv_full_x(res_qfrc, full_x)

    print(res_dfdx[0].reshape((d_params.n_state, d_params.n_state)))
    print(res_dfdu[0].reshape((d_params.n_state, d_params.n_ctrl)))
    print(res_dfinvdx[0].reshape((d_params.n_vel, d_params.n_state + d_params.n_vel)))
    print(res_dfdt[0])
    print(res_qfrc[0])
    print("DONE!")


