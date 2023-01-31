import torch
from models import DoubleCartpole, ModelParams
from animations.cartpole import animate_double_cartpole, init_fig_dcp
import numpy as np
from torch_mppi import MPPIController, MPPIParams

if __name__ == "__main__":
    dcp_params = ModelParams(3, 3, 1, 6, 6)
    pi_params = MPPIParams(150, 500, .5, 0.16, 0.01, 1.05)
    cp_mppi = DoubleCartpole(pi_params.K, dcp_params, 'cpu', mode='norm')
    cp_anim = DoubleCartpole(1, dcp_params, 'cpu', mode='norm')

    Q = torch.diag(torch.Tensor([.5, .5, .5, 0.0001, 0.0001, 0.0001])).repeat(pi_params.K, 1, 1).to('cpu')
    R = torch.diag(torch.Tensor([1/pi_params.sigma])).repeat(pi_params.K, 1, 1).to('cpu')
    Qs = torch.diag(torch.Tensor([.5, .2, .2, 0.0001, 0.0001, 0.0001]))

    def state_encoder(x: torch.Tensor):
        b, r, c = x.shape
        x = x.reshape((b, r * c))
        qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1:dcp_params.nq].clone(), x[:, dcp_params.nq:].clone()
        qp = torch.cos(qp) - 1
        return torch.cat((qc, qp, v), 1).reshape((b, r, c))

    u_cost = lambda u, u_pert: u @ R @ u_pert.mT
    r_cost = lambda x: state_encoder(x) @ Q @ state_encoder(x).mT
    t_cost = lambda x: r_cost(x)
    pi = MPPIController(cp_mppi, r_cost, t_cost, u_cost, pi_params)
    x = torch.Tensor([0, 3.14, 3.14, 0, 0, 0]).view(1, 1, 6)
    cart, pole1, pole2, us = [0, 0], [0, 0], [0, 0], []

    fig, p, r, width, height = init_fig_dcp(0)
    fig.show()

    for i in range(10000):
        cart[0], pole1[0], pole2[0] = x[:, :, 0].item(), x[:, :, 1].item(), x[:, :, 2].item()
        u = pi.MPC(x)
        xd = cp_anim(x, u)
        x = x + xd * 0.01
        print(f"x: {x}, u: {u}, cst: {x@Qs@x.mT}")
        us.append(u.item())
        cart[1], pole1[1], pole2[1] = x[:, :, 0].item(), x[:, :, 1].item(), x[:, :, 2].item()
        animate_double_cartpole(np.array(cart), np.array(pole1), np.array(pole2), fig, p, r, width, height)

