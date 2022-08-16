import numpy as np
from mujoco import derivative
from utilities.mj_utils import MjBatchOps
from collections import namedtuple
import mujoco
import torch
import cProfile

m = mujoco.MjModel.from_xml_path(
    "/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml"
)

batch_size = 2
DataParams = namedtuple('DataParams', 'n_full_state, n_state, n_pos, n_vel, n_ctrl, n_desc, n_batch')
d_params = DataParams(6, 4, 2, 2, 1, 2, batch_size)
d = mujoco.MjData(m)
bo = MjBatchOps(m, d_params)


def main():

    x = torch.zeros(batch_size, d_params.n_state)
    full_x = torch.zeros(batch_size, d_params.n_state + d_params.n_vel)
    u = torch.zeros(batch_size, d_params.n_ctrl)

    res_dfdx = bo.b_dfdx(x, u)
    res_dfdu = bo.b_dfdu(x, u)
    res_dfinvdx = bo.b_dfinvdx_full(full_x)
    res_dfdt = bo.b_dxdt(x, u)
    res_qfrc = bo.b_qfrcs(full_x)

    print(res_dfdx[0].reshape((d_params.n_state, d_params.n_state)))
    print(res_dfdu[0].reshape((d_params.n_state, d_params.n_ctrl)))
    print(res_dfinvdx[0].reshape((d_params.n_vel, d_params.n_state + d_params.n_vel)))
    print(res_dfdt[0])
    print(res_qfrc[0])
    print("DONE!")


if __name__ == "__main__":
    cProfile.run('main()')


