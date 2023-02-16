import torch
import matplotlib.pyplot as plt
from animations.cartpole import animate_cartpole
import numpy as np
from dataclasses import  dataclass

@dataclass
class ModelParams:
    nq: int = 0
    nv: int = 0
    nu: int = 0
    nx: int = 0
    nxd: int = 0


class BaseRBD(object):
    def __init__(self, nsims, params: ModelParams, device):
        self._params = params
        self._gear = 1
        self._b = torch.diag(torch.ones(params.nv)).repeat(nsims, 1, 1).to(device)

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

    def simulate(self, x, acc):
        q, qd = x[:, :, :self._params.nq].clone(), x[:, :, self._params.nq:].clone()
        M = self._Mfull(q)
        M_21, M_22 = M[:, 1, 0].unsqueeze(1).clone(), M[:, 1, 1].unsqueeze(1).clone()
        Tp = (self._Tgrav(q) - (self._Cfull(x) @ qd.mT).mT)[:, :, 1].clone()
        Tfric = (qd).mT[:, 1, :].clone() * self.FRICTION
        qddc = acc[:, :, 0].clone()
        qddp = 1/M_22 * (Tp - Tfric - M_21 * qddc)
        xd = torch.cat((qd[:, :, 0], qd[:, :, 1], qddc, qddp), 1).unsqueeze(1).clone()
        return xd



class Cartpole(BaseRBD):
    LENGTH = 1
    MASS_C = 1
    MASS_P = 1
    GRAVITY = -9.81
    FRICTION = .1
    GEAR = 2

    def __init__(self, nsims, params: ModelParams, device, augmented_state=False):
        super(Cartpole, self).__init__(nsims, params, device)
        self._L = torch.ones((nsims, 1, 1)).to(device) * self.LENGTH
        self._Mp = torch.ones((nsims, 1, 1)).to(device) * self.MASS_P
        self._Mc = torch.ones((nsims, 1, 1)).to(device) * self.MASS_C
        self._b *= self.FRICTION
        self._gear *= self.GEAR

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

    def _Tfric(self, v):
        return v * self.FRICTION

    def __call__(self, x, acc):
        return self.simulate(x, acc)


if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    cp = Cartpole(1, cp_params, 'cpu')
    x_init = torch.randn((1, 1, 4))
    xd_init = torch.randn((1, 1, 4))

    def test_func(x, xd):
        qc, qp, qd, qddc = x[:, :, 0], x[:, :, 1],  x[:, :, 2:], xd[:, :, 2]
        return torch.cat((qd[:, :, 0], qd[:, :, 1], qddc, -qddc * torch.cos(qp) - torch.sin(qp)), 1)

    ref = test_func(x_init, xd_init)
    res = cp(x_init, xd_init)
    print(torch.square(ref - res))

    x_init = torch.Tensor([0, .1, 0, 0]).view(1, 1, 4)
    xd_init = torch.Tensor([0, 0]).view(1, 1, 2)

    def integrate(func, x, xd, time, dt):
        xs = []

        for t in range(time):
            xd_new = func(x, xd)
            x = x + xd_new * dt
            xs.append(x)

        return xs


    xs = integrate(cp, x_init, xd_init, 500, 0.01)

    theta = [x[:, :, 1].item() for x in xs]
    cart = [x[:, :, 0].item() for x in xs]
    # plt.plot(theta)
    # plt.show()
    animate_cartpole(np.array(cart), np.array(theta))
