import torch
import matplotlib.pyplot as plt
from animations.cartpole import animate_cartpole, init_fig
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelParams:
    nq: int = 0
    nv: int = 0
    nu: int = 0
    nx: int = 0
    nxd: int = 0


class BaseRBD(object):
    def __init__(self, nsims, params: ModelParams, device, mode='pfl'):
        self._params = params
        self.simulator = self.simulate_REG
        if mode == 'pfl':
            self.simulator = self.simulate_PFL

    def _Muact(self, q):
        pass

    def _Mact(self, q):
        pass

    def _Mfull(self, q):
        pass

    def _Cfull(self, x):
        pass

    def _Tgrav(self, q):
        pass

    def _Bvec(self):
        pass

    def _Tfric(self, qd):
        pass

    def simulate_PFL(self, x, acc):
        q, qd = x[:, :, :self._params.nq].clone(), x[:, :, self._params.nq:].clone()
        M = self._Mfull(q)
        M_21, M_22 = M[:, 1, 0].unsqueeze(1).clone(), M[:, 1, 1].unsqueeze(1).clone()
        Tp = (self._Tgrav(q) - (self._Cfull(x) @ qd.mT).mT)[:, :, 1].clone()
        Tfric = self._Tfric(qd)[:, :, 1].clone()
        qddc = acc[:, :, 0].clone()
        qddp = 1/M_22 * (Tp - Tfric - M_21 * qddc)
        xd = torch.cat((qd[:, :, 0], qd[:, :, 1], qddc, qddp), 1).unsqueeze(1).clone()
        return xd

    def simulate_REG(self, x, tau):
        tau = tau.clamp(-1, 1)
        q, qd = x[:, :, :self._params.nq].clone(), x[:, :, self._params.nq:].clone()
        Minv = torch.linalg.inv(self._Mfull(q))
        Tp = (self._Tgrav(q) - (self._Cfull(x) @ qd.mT).mT)
        Tfric = self._Tfric(qd)
        B = self._Bvec()
        qdd = (Minv @ (Tp - Tfric + B * tau).mT).mT
        xd = torch.cat((qd[:, :, 0:2], qdd), 2).clone()
        return xd


class Cartpole(BaseRBD):
    LENGTH = 1
    MASS_C = 1
    MASS_P = 1
    GRAVITY = -9.81
    FRICTION = .1
    GEAR = 30

    def __init__(self, nsims, params: ModelParams, device, mode='pfl'):
        super(Cartpole, self).__init__(nsims, params, device, mode)
        self._L = torch.ones((nsims, 1, 1)).to(device) * self.LENGTH
        self._Mp = torch.ones((nsims, 1, 1)).to(device) * self.MASS_P
        self._Mc = torch.ones((nsims, 1, 1)).to(device) * self.MASS_C
        self._b = torch.Tensor([1, 0]).repeat(nsims, 1, 1).to(device)

    def _Muact(self, q):
        qc, qp = q[:, :, 0].unsqueeze(1).clone(), q[:, :, 1].unsqueeze(1).clone()
        M21 = (self.MASS_P * self.LENGTH * torch.cos(qp))
        M22 = (self.MASS_P * self.LENGTH ** 2) * torch.ones_like(qp)
        # self._M[:, 1, 0] = (self._Mp * self._L * torch.cos(qp)).squeeze()
        # self._M[:, 1, 1] = (self._Mp * self._L ** 2).squeeze()
        return torch.cat((M21, M22), 2)
        # return self._M[:, 1, :].clone()

    def _Mact(self, q):
        qc, qp = q[:, :, 0].unsqueeze(1).clone(), q[:, :, 1].unsqueeze(1).clone()
        M00 = (self.MASS_P + self.MASS_C) * torch.ones_like(qp)
        M01 = (self.MASS_P * self.LENGTH * torch.cos(qp))
        # self._M[:, 0, 0] = (self._Mp + self._Mc).squeeze()
        # self._M[:, 0, 1] = (self._Mp * self._L * torch.cos(qp)).squeeze()
        # return self._M[:, 0, :].clone()
        return torch.cat((M00, M01), 2)

    def _Mfull(self, q):
        Mtop = self._Mact(q)
        Mlow = self._Muact(q)
        return torch.hstack((Mtop, Mlow))
        # return self._M.clone()

    def _Cfull(self, x):
        qp, qdp = x[:, :, 1].unsqueeze(1).clone(), x[:, :, 3].unsqueeze(1).clone()
        C12 = (-self.MASS_P * self.LENGTH * qdp * torch.sin(qp))
        Ctop = torch.cat((torch.zeros_like(C12), C12), 2)
        return torch.cat((Ctop, torch.zeros_like(Ctop)), 1)

    def _Tgrav(self, q):
        qc, qp = q[:, :, 0].unsqueeze(1), q[:, :, 1].unsqueeze(1)
        grav = (-self.MASS_P * self.GRAVITY * self.LENGTH * torch.sin(qp))
        return torch.cat((torch.zeros_like(grav), grav), 2)

    def _Bvec(self):
        return self._b * self.GEAR

    def _Tfric(self, qd):
        return qd * self.FRICTION

    def __call__(self, x, inputs):
        return self.simulator(x, inputs)


if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    cp = Cartpole(1, cp_params, 'cpu', mode='norm')

    # x_init = torch.randn((1, 1, 4))
    # qdd_init = torch.randn((1, 1, 4))
    # def test_func(x, xd):
    #     qc, qp, qd, qddc = x[:, :, 0], x[:, :, 1],  x[:, :, 2:], xd[:, :, 2]
    #     return torch.cat((qd[:, :, 0], qd[:, :, 1], qddc, -qddc * torch.cos(qp) - torch.sin(qp)), 1)
    #
    # ref = test_func(x_init, qdd_init)1,
    # res = cp(x_init, qdd_init)
    # print(torch.square(ref - res))

    x_init = torch.Tensor([0, 3.14, 0, 0]).view(1, 1, 4)
    qdd_init = torch.Tensor([0, 0]).view(1, 1, 2)


    def integrate(func, x, xd, time, dt):
        xs = []
        xs.append(x)
        for t in range(time):
            xd_new = func(x, torch.randn((1,1)) * 10)
            x = x + xd_new * dt
            xs.append(x)

        return xs


    xs = integrate(cp, x_init, qdd_init, 5000, 0.01)

    theta = [x[:, :, 1].item() for x in xs]
    cart = [x[:, :, 0].item() for x in xs]
    plt.plot(cart)
    plt.show()
    fig, p, r, width, height = init_fig(cart[0])
    animate_cartpole(np.array(cart), np.array(theta), fig, p, r, width, height)