import torch
import matplotlib.pyplot as plt
from utilities.torch_device import device
from animations.cartpole import animate_double_cartpole, init_fig_dcp
from animations.cartpole import animate_cartpole, init_fig_cp
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

    def _Tbias(self, x):
        pass

    def _Bvec(self):
        pass

    def _Tfric(self, qd):
        pass

    def inverse_dynamics(self, x, acc):
        q, qd = x[:, :, :self._params.nq].clone(), x[:, :, self._params.nq:].clone()
        M = self._Mfull(q)
        Tp = self._Tbias(x)
        Tfric = self._Tfric(qd)
        M_21, M_22 = M[:, 1, 0].unsqueeze(1).clone(), M[:, 1, 1].unsqueeze(1).clone()
        qddc = acc[:, :, 0].clone()
        qddp = 1/M_22 * (Tp[:, :, 1].clone() - Tfric[:, :, 1].clone() - M_21 * qddc)
        acc = torch.cat((qddc, qddp), dim=1).unsqueeze(0)
        T = (M @ acc.mT).mT - Tp + Tfric
        return T

    def PFL(self, x, acc):
        pass

    def simulate_PFL(self, x, acc):
        return self.PFL(x, acc)

    def simulate_REG(self, x, tau):
        q, qd = x[:, :, :self._params.nq].clone(), x[:, :, self._params.nq:].clone()
        Minv = torch.linalg.inv(self._Mfull(q))
        Tp = self._Tbias(x)
        Tfric = self._Tfric(qd)
        B = self._Bvec()
        qdd = (Minv @ (Tp + Tfric + B * tau).mT).mT
        xd = torch.cat((qd[:, :, 0:self._params.nx], qdd), 2).clone()
        return xd

class DoubleIntegrator(BaseRBD):
    MASS = 1
    FRICTION = .1
    GEAR = 1

    def __init__(self, nsims, params: ModelParams, device, mode='pfl'):
        super(DoubleIntegrator, self).__init__(nsims, params, device, mode)
        self._M = torch.ones((nsims, 1, 1)).to(device) * self.MASS
        self._b = torch.Tensor([1]).repeat(nsims, 1, 1).to(device)

    def _Muact(self, q):
        return None

    def _Mact(self, q):
        return self.MASS * torch.ones_like(q)

    def _Mfull(self, q):
        return self.MASS * torch.ones_like(q)

    def _Mu_Mua(self, q):
        return None, self._M

    def _Cfull(self, x):
        return self.MASS * torch.ones_like(x[:,:,0].unsqueeze(2))

    def _Tgrav(self, q):
        return torch.ones_like(q) * 0

    def _Tbias(self, x):
        return torch.ones_like(x[:,:,0].unsqueeze(2)) * 0

    def _Bvec(self):
        return self._b * self.GEAR

    def _Tfric(self, qd):
        return qd * self.FRICTION

    def __call__(self, x, inputs):
        return self.simulator(x, inputs)

    def PFL(self, x, acc):
        xd = torch.cat((x[:, :, -1].unsqueeze(2), acc), 2).clone()
        return xd


class Cartpole(BaseRBD):
    LENGTH = 1
    MASS_C = 1
    MASS_P = 1
    GRAVITY = -9.81
    FRICTION = torch.Tensor([0.1, 0.1]).to(device)
    GEAR = 60

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

    def _Mu_Mua(self, q):
        M = self._Mfull(q)
        Mu, Mua = M[:, 1, 1].clone().view(q.shape[0], 1, 1), M[:, 0, 1].clone().view(q.shape[0], 1, 1)
        return Mu, Mua

    def _Cfull(self, x):
        qp, qdp = x[:, :, 1].unsqueeze(1).clone(), x[:, :, 3].unsqueeze(1).clone()
        C12 = (-self.MASS_P * self.LENGTH * qdp * torch.sin(qp))
        Ctop = torch.cat((torch.zeros_like(C12), C12), 2)
        return torch.cat((Ctop, torch.zeros_like(Ctop)), 1)

    def _Tgrav(self, q):
        qc, qp = q[:, :, 0].unsqueeze(1), q[:, :, 1].unsqueeze(1)
        grav = (-self.MASS_P * self.GRAVITY * self.LENGTH * torch.sin(qp))
        return torch.cat((torch.zeros_like(grav), grav), 2)

    def _Tbias(self, x):
        q, qd = x[:, :, :2].clone(), x[:, :, 2:].clone()
        return self._Tgrav(q) - (self._Cfull(x) @ qd.mT).mT

    def _Bvec(self):
        return self._b * self.GEAR

    def _Tfric(self, qd):
        return qd * self.FRICTION

    def __call__(self, x, inputs):
        return self.simulator(x, inputs)

    def PFL(self, x, acc):
        q, qd = x[:, :, :self._params.nq].clone(), x[:, :, self._params.nq:].clone()
        M = self._Mfull(q)
        M_21, M_22 = M[:, 1, 0].unsqueeze(1).clone(), M[:, 1, 1].unsqueeze(1).clone()
        Tp = (self._Tgrav(q) - (self._Cfull(x) @ qd.mT).mT)[:, :, 1].clone()
        Tfric = self._Tfric(qd)[:, :, 1].clone()
        qddc = acc[:, :, 0].clone()
        qddp = 1/M_22 * (Tp - Tfric - M_21 * qddc)
        xd = torch.cat((qd[:, :, 0], qd[:, :, 1], qddc, qddp), 1).unsqueeze(1).clone()
        return xd


class DoubleCartpole(BaseRBD):
    LENGTH = 1
    MASS_C = 1
    MASS_P = 1
    GRAVITY = 9.81
    FRICTION = .01
    GEAR = 30

    def __init__(self, nsims, params: ModelParams, device, mode='pfl'):
        super(DoubleCartpole, self).__init__(nsims, params, device, mode)
        self._b = torch.Tensor([1, 0, 0]).repeat(nsims, 1, 1).to(device)

    def _Mact(self, q):
        return self._Ms(q)[1]

    def _Muact(self, q):
        return self._Ms(q)[2]

    def _Mfull(self, q):
        return self._Ms(q)[0]

    def _Ms(self, q):
        qc, qp1, qp2 = q[:, :, 0].unsqueeze(1).clone(), q[:, :, 1].unsqueeze(1).clone(), q[:, :, 2].unsqueeze(1).clone()
        M11 = (self.MASS_P + self.MASS_P + self.MASS_C) * torch.ones_like(qc)
        M12 = -0.5 * (self.MASS_P + self.MASS_P) * self.LENGTH * torch.cos(qp1)
        M13 = -0.5 * self.MASS_P * self.LENGTH * torch.cos(qp2)
        M22 = (self.MASS_P * self.LENGTH ** 2 + (1/12 * self.LENGTH * self.MASS_P) + 0.25 * self.MASS_P * self.LENGTH ** 2) * torch.ones_like(qc)
        M23 = 0.5 * self.MASS_P * self.LENGTH ** 2 * torch.cos((qp1 - qp2))
        M33 = (0.25 * self.MASS_P * self.LENGTH ** 2 + (1/12 * self.LENGTH * self.MASS_P) ) * torch.ones_like(qc)

        M1s = torch.cat((M11, M12, M13), dim=2)
        M2s = torch.cat((M13, M22, M23), dim=2)
        M3s = torch.cat((M13, M23, M33), dim=2)

        Mact = torch.hstack(
            (torch.cat((M12, torch.zeros_like(M12)), dim=2), torch.cat((torch.zeros_like(M13), M13), dim=2))
        )

        Muact = torch.hstack(
            (torch.cat((M22, M23), dim=2), torch.cat((M23, M33), dim=2))
        )

        Mfull = torch.hstack(
            (M1s, M2s, M3s)
        )

        return Mfull, Mact, Muact

    def _Mu_Mua(self, q):
        M = self._Mfull(q)
        Mu, Mua = M[:, 1:, 1:].clone().view(q.shape[0], 2, 2), M[:, 1:, 0].clone().view(q.shape[0], 1, 2)
        return Mu, Mua

    def _Cfull(self, x):
        pass

    def _Tgrav(self, q):
        pass

    def _Tbias(self, x):
        qc, qp1, qp2 = x[:, :, 0].unsqueeze(1).clone(), x[:, :, 1].unsqueeze(1).clone(), x[:, :, 2].unsqueeze(1).clone()
        qdc, qdp1, qdp2 = x[:, :, 3].unsqueeze(1).clone(), x[:, :, 4].unsqueeze(1).clone(), x[:, :, 5].unsqueeze(1).clone()

        L, fric, G = self.LENGTH, self.FRICTION, self.GRAVITY

        Tact = - 0.5 * (self.MASS_P + 2*self.MASS_P) * L * qdp1**2 *torch.sin(qp1) - 0.5 * self.MASS_P * L * qdp2**2 * torch.sin(qp2)
        T2 = (0.5 * self.MASS_P + self.MASS_P) * L * G * torch.sin(qp1) - 0.5 * self.MASS_P * L**2 * qdp2**2 * torch.sin((qp1 - qp2))
        T3 = (0.5 * self.MASS_P) * L * G * torch.sin(qp2) + L**2 * qdp1**2 * torch.sin((qp1 - qp2))

        return torch.cat((Tact, T2, T3), dim=2)

    def _Bvec(self):
        return self._b * self.GEAR

    def _Tfric(self, qd):
        return qd * self.FRICTION

    def PFL(self, x, acc):
        q, qd = x[:, :, :self._params.nq].clone(), x[:, :, self._params.nq:].clone()
        qddc = acc[:, :, 0].clone().unsqueeze(1)
        qddc_vec = torch.cat((qddc, qddc), dim=2)
        Ta = (self._Mact(q) @ qddc_vec.mT).mT
        Tu = self._Tbias(x)[:, :, 1:].clone()
        Tfric = self._Tfric(qd)[:, :, 1:].clone()
        qddps = (torch.linalg.inv(self._Muact(q)) @ (Tu - Tfric - Ta).mT).mT
        return torch.cat((qd, qddc, qddps), dim=2)

    def __call__(self, x, inputs):
        return self.simulator(x, inputs)


if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    cp = Cartpole(1, cp_params, 'cpu', mode='pfl')
    dcp_params = ModelParams(3, 3, 1, 6, 6)
    dcp = DoubleCartpole(1, dcp_params, 'cpu', mode='pfl')
    x_init_cp = torch.Tensor([0, torch.pi-0.3, 0, 0]).view(1, 1, 4)
    qdd_init_cp = torch.Tensor([0, 0]).view(1, 1, 2)
    x_init_dcp = torch.Tensor([0, 2, 2, 0, 0, 0]).view(1, 1, 6)
    qdd_init_dcp = torch.Tensor([0, 0, 0]).view(1, 1, 3)
    ctrl = np.load('ctrl.npy')
    ctrl = torch.Tensor(ctrl)

    def integrate(func, x, xd, time, dt):
        xs = []
        xs.append(x)
        for t in range(time):
            xd_new = func(x, ctrl[t])
            x = x + xd_new * dt
            xs.append(x)

        return xs


    xs_cp = integrate(cp.simulate_REG, x_init_cp, qdd_init_cp, 500, 0.01)
    # xs_dcp = integrate(dcp, x_init_dcp, qdd_init_dcp, 5000, 0.01)

    theta1 = [x[:, :, 1].item() for x in xs_cp]
    # theta2 = [x[:, :, 2].item() for x in xs_dcp]
    cart = [x[:, :, 0].item() for x in xs_cp]

    plt.plot(cart)
    plt.show()
    fig, p, r, width, height = init_fig_cp(cart[0])
    animate_cartpole(np.array(cart), np.array(theta1), fig, p, r, width, height, skip=2)