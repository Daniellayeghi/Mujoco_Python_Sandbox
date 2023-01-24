import torch
from dataclasses import dataclass
from models import Cartpole, ModelParams
from animations.cartpole import animate_cartpole, init_fig
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class MPPIParams:
    T: int = 0
    K: int = 0
    temp: float = 0
    sigma: float = 0
    delta: float = 0.01
    discount: float = 1.01


class MPPIController:
    def __init__(self, dynamics: Cartpole, r_cost, t_cost, u_cost, params: MPPIParams):
        self._dynamics = dynamics
        self._r_cost = r_cost
        self._t_cost = t_cost
        self._u_cost = u_cost
        self._params = params
        self._states = torch.empty((self._params.T+1, self._params.K, 1, self._dynamics._params.nx))
        self._perts = torch.empty(((self._params.T, self._params.K, 1, self._dynamics._params.nu)))
        self._new_u = torch.empty(((self._params.T, self._params.K, 1, self._dynamics._params.nu)))
        self._ctrls = torch.zeros(((self._params.T, 1, self._dynamics._params.nu)))
        self._costs = torch.zeros(((self._params.K)))
        self._norm = None

    def sample_ctrl(self):
        self._perts = torch.randn_like(self._perts) * self._params.sigma
        return self._perts

    def rollout(self, x):
        disc = 1
        self._costs *= 0
        x_init = x.clone().repeat(self._params.K, 1, 1)
        self._perts = self.sample_ctrl()
        self._states[0, :, :, :] = x_init
        self._costs += self._r_cost(x_init).squeeze()
        for t in range(self._params.T):
            self._new_u[t, :, :, :] = torch.clamp(self._perts[t, :, :, :] + self._ctrls[t, :, :], -1.0, 1.0)
            xd = self._dynamics(x_init, self._new_u[t, :, :, :])
            x_init = x_init + xd * self._params.delta
            self._states[t+1, :, :, :] = x_init
            u_reg = pi_params.temp * self._u_cost(self._ctrls[t, :, :], self._perts[t, :, :, :])
            self._costs += self._r_cost(x_init).squeeze() * disc + u_reg.squeeze()
            disc *= self._params.discount

        self._costs += self._t_cost(self._states[-1, :, :, :]).squeeze() * (disc * self._params.discount)

    def compute_ctrls(self):
        min_cst = self._costs.min()
        norm = torch.sum(torch.exp(-1 / self._params.temp * (self._costs - min_cst)), 0).squeeze()
        weights = torch.exp(-1 / self._params.temp * (self._costs - min_cst))/norm
        self._norm = weights

        for t in range(self._params.T):
            ctrl = self._ctrls[t, :, :]
            ctrl += torch.sum(weights * self._perts[t, :, :, :].squeeze())

        self._ctrls = self._ctrls.clamp(-1, 1)
        return self._ctrls

    def MPC(self, x):
        self.rollout(x)
        self._ctrls = self.compute_ctrls()
        u = self._ctrls[0, :, :]
        self._ctrls = self._ctrls.roll(-1, 0)
        self._ctrls[-1, :, :] *= 0
        return u


if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    pi_params = MPPIParams(75, 200, .5, 0.16, 0.01, 1.1)
    cp_mppi = Cartpole(pi_params.K, cp_params, 'cpu', mode='norm')
    cp_anim = Cartpole(1, cp_params, 'cpu', mode='norm')

    Q = torch.diag(torch.Tensor([2, 2, 0.01, 0.01])).repeat(pi_params.K, 1, 1).to('cpu')
    Qf = torch.diag(torch.Tensor([2, 2, 0.01, 0.01])).repeat(pi_params.K, 1, 1).to('cpu')
    R = torch.diag(torch.Tensor([1/pi_params.sigma])).repeat(pi_params.K, 1, 1)

    def state_encoder(x: torch.Tensor):
        b, r, c = x.shape
        x = x.reshape((b, r * c))
        qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
        qp = torch.cos(qp) - 1
        return torch.cat((qc, qp, v), 1).reshape((b, r, c))

    u_cost = lambda u, u_pert: u @ R @ u_pert.mT
    r_cost = lambda x: x @ Q @ x.mT
    t_cost = lambda x: x @ Qf @ x.mT

    pi = MPPIController(cp_mppi, r_cost, t_cost, u_cost, pi_params)
    x = torch.Tensor([0, 3.14, 0, 0]).view(1, 1, 4)
    cart, pole, us = [0, 0], [0, 0], []

    fig, p, r, width, height = init_fig(0)
    fig.show()

    for i in range(1000):
        cart[0], pole[0] = x[:, :, 0].item(), x[:, :, 1].item()
        u = pi.MPC(x)
        xd = cp_anim(x, u)
        x = x + xd * 0.01
        print(f"x: {x}, u: {u}")
        us.append(u.item())
        cart[1], pole[1] = x[:, :, 0].item(), x[:, :, 1].item()
        animate_cartpole(np.array(cart), np.array(pole), fig, p, r, width, height)

