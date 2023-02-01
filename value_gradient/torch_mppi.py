import torch
from dataclasses import dataclass

@dataclass
class MPPIParams:
    T: int = 0
    K: int = 0
    temp: float = 0
    sigma: float = 0
    delta: float = 0.01
    discount: float = 1.01


class MPPIController:
    def __init__(self, dynamics, r_cost, t_cost, u_cost, params: MPPIParams):
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
            u_reg = self._params.temp * self._u_cost(self._ctrls[t, :, :], self._perts[t, :, :, :])
            self._costs += self._r_cost(x_init).squeeze() * disc + u_reg.squeeze()
            disc *= self._params.discount

        self._costs += self._t_cost(self._states[-1, :, :, :]).squeeze() * (disc * self._params.discount)
        print(torch.mean(self._costs))

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

