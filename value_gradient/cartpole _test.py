import torch
from models import Cartpole, ModelParams
from animations.cartpole import animate_cartpole, init_fig_cp
import numpy as np
from torch_mppi import MPPIController, MPPIParams
from mj_renderer import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    pi_params = MPPIParams(10, 200, .5, 0.16, 0.01, 1.1)
    cp_mppi = Cartpole(pi_params.K, cp_params, 'cpu', mode='norm')
    cp_anim = Cartpole(1, cp_params, 'cpu', mode='norm')
    ren = MjRenderer("../xmls/cartpole.xml")
    Q = torch.diag(torch.Tensor([.05, 5, .1, .1])).repeat(pi_params.K, 1, 1).to('cpu')
    Qf = torch.diag(torch.Tensor([1, 10000, .001, 10])).repeat(pi_params.K, 1, 1).to('cpu')
    Qf_single = torch.diag(torch.Tensor([0, 2, 0, 0.1])).repeat(1, 1, 1).to('cpu')
    R = torch.diag(torch.Tensor([1/pi_params.sigma])).repeat(pi_params.K, 1, 1).to('cpu')
    value = torch.jit.load('HJB_value.pt')

    def state_encoder(x: torch.Tensor):
        b, r, c = x.shape
        x = x.reshape((b, r * c))
        qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
        qp = torch.cos(qp) - 1
        return torch.cat((qc, qp, v), 1).reshape((b, r, c))

    u_cost = lambda u, u_pert: u @ R @ u_pert.mT
    r_cost = lambda x: state_encoder(x) @ Q @ state_encoder(x).mT
    t_cost = lambda x: value.forward(torch.zeros((1)), (state_encoder(x))) * 0.4
    t_reg_cost = lambda x: state_encoder(x) @ Qf @ state_encoder(x).mT

    pi = MPPIController(cp_mppi, r_cost, t_reg_cost, u_cost, pi_params)
    x = torch.Tensor([0, 3.14, 0, 0]).view(1, 1, 4)

    fig, p, r, width, height = init_fig_cp(0)
    fig.show()

    for i in range(10000):
        u = pi.MPC(x)
        xd = cp_anim(x, u)
        x = x + xd * 0.01

        if torch.abs(state_encoder(x)[:, :, 1]).item() < 0.012:
            print("STABALIZING")
            pi._r_cost = t_reg_cost
            pi._t_cost = t_reg_cost
        else:
            pi._t_cost = t_cost

        print(f"x: {x}, u: {u}")
        ren.render(x[:, :, :cp_params.nq].clone().detach().numpy())
        # us.append(u.item())
        # cart[1], pole[1] = x[:, :, 0].item(), x[:, :, 1].item()
        # animate_cartpole(np.array(cart), np.array(pole), fig, p, r, width, height)

