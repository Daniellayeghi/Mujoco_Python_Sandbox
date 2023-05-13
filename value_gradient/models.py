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
    def __init__(self, nsims, params: ModelParams, device, mode, stabilize=False):
        self._params = params
        self.simulator = self.simulate_REG
        self._mode = mode
        if mode == 'inv':
            self.simulator = self.simulate_PFL
        if mode == 'fwd':
            self.simulator = self.simulate_REG
        self._stabilize = stabilize


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

    def _FWDreg(self, x):
        pass

    def _INVreg(self, x):
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
        qdd = (Minv @ (Tp - Tfric + B*tau).mT).mT
        xd = torch.cat((qd[:, :, 0:self._params.nx], qdd), 2).clone()
        return xd


class DoubleIntegrator(BaseRBD):
    MASS = 1
    FRICTION = .1
    GEAR = 1

    def __init__(self, nsims, params: ModelParams, device, mode='inv'):
        super(DoubleIntegrator, self).__init__(nsims, params, device, mode)
        self._M = torch.ones((nsims, 1, 1)).to(device) * self.MASS
        self._b = torch.diag(torch.Tensor([1])).repeat(nsims, 1, 1).to(device)

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
    LENGTH = .3
    MASS_C = 1
    MASS_P = 0.1
    GRAVITY = -9.81
    FRICTION = torch.Tensor([0.0, 0.1]).to(device)
    GEAR = 1

    def __init__(self, nsims, params: ModelParams, device, mode='inv', stabalize=False):
        super(Cartpole, self).__init__(nsims, params, device, mode, stabalize)
        self._L = torch.ones((nsims, 1, 1)).to(device) * self.LENGTH
        self._Mp = torch.ones((nsims, 1, 1)).to(device) * self.MASS_P
        self._Mc = torch.ones((nsims, 1, 1)).to(device) * self.MASS_C
        self._b = torch.Tensor([1, 0]).repeat(nsims, 1, 1)
        self.K = torch.Tensor([5.162, 40.269, 0.05, .4]).repeat(nsims, 1, 1).to(device)

        def state_encoder(x: torch.Tensor):
            b, r, c = x.shape
            x = x.reshape((b, r * c))
            qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
            qp = torch.cos(qp) - 1
            return torch.cat((qc, qp, v), 1).reshape((b, r, c))
        self._state_encoder = state_encoder

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

    def _FWDreg(self, x):
        # x_enc = self._state_encoder(x)
        reg = x @ self.K.mT
        return torch.cat((reg, torch.zeros_like(reg)), dim=2).to(device)

    def _INVreg(self, x):
        # x = self._state_encoder(x)
        return x @ self.K

    def g(self, x):
        q, qd = x[:, :, :2].clone(), x[:, :, 2:].clone()
        Mu, Mua = self._Mu_Mua(q)
        B = -torch.inverse(Mua) @ Mua
        zeros = torch.zeros_like(qd).to(device)
        ones = torch.ones((qd.shape[0], 1, 1)).to(device)
        return torch.cat((zeros, B, ones), dim=2)


    def __call__(self, x, inputs):
        if self._stabilize:
            stab = (torch.abs(self._state_encoder(x)[:, :, 1]) < 0.17).squeeze()
            not_stab = (torch.abs(self._state_encoder(x)[:, :, 1]) > 0.17).squeeze()

            ctrl = self._FWDreg(x) if self._mode != "inv" else self._INVreg(x)
            inputs = inputs * not_stab.reshape(x.shape[0], 1, 1) + ctrl * stab.reshape(x.shape[0], 1, 1)

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
    MASS_P = 0.1
    GRAVITY = 9.81
    FRICTION = 0.001
    GEAR = 30

    def __init__(self, nsims, params: ModelParams, device, mode='inv'):
        super(DoubleCartpole, self).__init__(nsims, params, device, mode)
        self._b = torch.diag(torch.Tensor([1, 0, 0])).repeat(nsims, 1, 1).to(device)

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
        M22 = (self.MASS_P * self.LENGTH ** 2 + (1/12 * self.LENGTH**2 * self.MASS_P) + 0.25 * self.MASS_P * self.LENGTH ** 2) * torch.ones_like(qc)
        M23 = 0.5 * self.MASS_P * self.LENGTH ** 2 * torch.cos((qp1 - qp2))
        M33 = (0.25 * self.MASS_P * self.LENGTH ** 2 + (1/12 * self.LENGTH**2 * self.MASS_P)) * torch.ones_like(qc)

        M1s = torch.cat((M11, M12, M13), dim=2)
        M2s = torch.cat((M12, M22, M23), dim=2)
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
        T3 = (0.5 * self.MASS_P * L * (G * torch.sin(qp2) + L**2 * qdp1**2 * torch.sin((qp1 - qp2))))

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


class TwoLink2(BaseRBD):
    FRICTION = 0.025
    GEAR = 1

    def __init__(self, nsims, params: ModelParams, device, mode='inv'):
        super(TwoLink2, self).__init__(nsims, params, device, mode)
        self._b = torch.Tensor([1, 1]).repeat(nsims, 1, 1).to(device)

    def _Mact(self, q):
        return self._Ms(q)[1]

    def _Muact(self, q):
        return self._Ms(q)[2]

    def _Mfull(self, q):
        return self._Ms(q)[0]

    def _Ms(self, q):
        qp1, qp2 = q[:, :, 0].unsqueeze(1).clone(), q[:, :, 1].unsqueeze(1).clone()
        ones = torch.ones_like(qp1)
        a1, a2, a3 = 0.025 + 0.045 + 1.4 * 0.3 ** 2, 0.3 * 0.16, 0.045
        M11 = a1 * ones + 2 * a2 * torch.cos(qp2)
        M12 = a3 * ones + a2 * torch.cos(qp2)
        M22 = a3 * ones
        M1s = torch.cat((M11, M12), dim=2)
        M2s = torch.cat((M12, M22), dim=2)

        Mfull = torch.hstack(
            (M1s, M2s)
        )

        return Mfull, Mfull, None

    def _Mu_Mua(self, q):
        return None, None

    def _Cfull(self, x):
        pass

    def _Tgrav(self, q):
        pass

    def _Tbias(self, x):
        qp1, qp2 = x[:, :, 0].unsqueeze(1).clone(), x[:, :, 1].unsqueeze(1).clone(),
        qdp1, qdp2 = x[:, :, 2].unsqueeze(1).clone(), x[:, :, 3].unsqueeze(1).clone()
        qd = x[:, :, 2:].unsqueeze(1).clone()
        a2 = 0.3 * 0.1
        T1act = -qdp2 * (2 * qdp1 + qdp2)
        T2act = qdp1 ** 2
        return -torch.cat((T1act, T2act), dim=2) * (a2 * torch.sin(qp2))

    def _Bvec(self):
        return self._b * self.GEAR

    def _Tfric(self, qd):
        return qd * self.FRICTION

    def PFL(self, x, acc):
        qd = x[:, :, 2:]
        xd = torch.cat((qd, acc), 2).clone()
        return xd

    def __call__(self, x, inputs):
        return self.simulator(x, inputs)


if __name__ == "__main__":
    from mj_renderer import *
    # ren = MjRenderer('../xmls/double_cart_pole.xml')
    ren_tl = MjRenderer('../xmls/reacher.xml')
    ren_cp = MjRenderer('../xmls/cartpole.xml')

    cp_params = ModelParams(2, 2, 1, 4, 4)
    cp = Cartpole(1, cp_params, 'cpu', mode='fwd', stabalize=False)

    dcp_params = ModelParams(3, 3, 1, 6, 6)
    dcp = DoubleCartpole(1, dcp_params, 'cpu', mode='inv')

    tl_params = ModelParams(2, 2, 2, 4, 4)
    tl = TwoLink2(1, tl_params, 'cpu', mode='fwd')

    x_init_cp = torch.Tensor([0, 0.7, 0, 0]).view(1, 1, 4) * 0
    qdd_init_cp = torch.Tensor([0, 0]).view(1, 1, 2)
    traj_cp = torch.zeros((10, 1, 1, cp_params.nx))

    x_init_dcp = torch.Tensor([0, 0.1, 0.1, 0, 0, 0]).view(1, 1, 6)
    qdd_init_dcp = torch.Tensor([0, 0, 0]).view(1, 1, 3)
    traj_dcp = torch.zeros((500, 1, 1, dcp_params.nx))

    x_init_tl = torch.Tensor([0, 0, 0, 0]).view(1, 1, 4)
    qdd_init_tl = torch.Tensor([0, 0]).view(1, 1, 2)
    traj_tl = torch.zeros((10, 1, 1, tl_params.nx))
    # test_acc = torch.from_numpy(np.load('test_acc.npy'))[:,:,:,0].reshape(200, 1, 1, 1)

    K = torch.Tensor([-1.162, -2.269, 0, 0]).reshape(1, 4, 1)
    np.random.seed(0)
    u = np.random.randn(10)
    u = torch.Tensor(u)


    def integrate(func, x, xd, time, dt, res: torch.Tensor):
        for t in range(time):
            xd_new = func(x, u[t])
            x = x + xd_new * dt
            res[t] = x

        return res

    #
    # def transform_coordinates_dcp(traj: torch.Tensor):
    #    # traj[:, :, :, 1] = traj[:, :, :, 1] + 2 * (torch.pi - traj[:, :, :, 1])
    #    traj[:, :, :, 2] = torch.pi - (traj[:, :, :, 1] + (torch.pi - traj[:, :, :, 2]))
    #    return traj
    #
    # # xs_cp = integrate(cp.simulate_REG, x_init_cp, qdd_init_cp, 500, 0.01)
    # xs_dcp = integrate(dcp, x_init_dcp, qdd_init_dcp, 500, 0.01, traj_dcp)
    # traj_dcp_mj = transform_coordinates_dcp(traj_dcp)
    # ren.render(traj_dcp_mj[:, 0, 0, :dcp_params.nq].cpu().detach().numpy())

    # theta1 = [x[:, :, 1].item() for x in xs_dcp]
    # theta2 = [x[:, :, 2].item() for x in xs_dcp]
    # cart = [x[:, :, 0].item() for x in xs_dcp]
    #
    # fig, p, r, width, height = init_fig_dcp(0)
    # animate_double_cartpole(np.array(cart), np.array(theta1), np.array(theta2), fig, p, r, width, height, skip=2)


    # def transform_coordinates_tl(traj: torch.Tensor):
    #    traj[:, :, :, 1] = torch.pi - (traj[:, :, :, 0] + (torch.pi - traj[:, :, :, 1]))
    #    return traj
    #
    # traj_tl = integrate(tl, x_init_tl, qdd_init_tl, 10, 0.01, traj_tl)
    # np.save('x_test_tl.npy', traj_tl.detach().cpu().numpy())
    # p =1
    # traj_tl_mj = transform_coordinates_tl(traj_tl)
    # ren_tl.render(traj_tl_mj[:, 0, 0, :tl_params.nq].cpu().detach().numpy())

    # np.save('u_test.npy', u.detach().cpu().numpy())
    traj_cp = integrate(cp, x_init_cp, qdd_init_cp, 10, 0.01, traj_cp)
    np.save('x_test.npy', traj_cp.detach().cpu().numpy())
    # ren_cp.render(traj_cp[:, 0, 0, :cp_params.nq].cpu().detach().numpy())