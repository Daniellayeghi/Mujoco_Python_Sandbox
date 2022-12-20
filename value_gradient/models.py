import torch
import matplotlib.pyplot as plt


class Cartpole:
    LENGTH = 1
    MASS_C = 1
    MASS_P = 1
    GRAVITY = -9.81
    FRICTION = .5

    def __init__(self, nsims):
        self._L = torch.ones((nsims, 1, 1)) * self.LENGTH
        self._Mp = torch.ones((nsims, 1, 1)) * self.MASS_P
        self._Mc = torch.ones((nsims, 1, 1)) * self.MASS_C
        self._Ma = torch.zeros((nsims, 1, 2))
        self._Mu = torch.zeros((nsims, 1, 2))
        self._M = torch.zeros((nsims, 2, 2))
        self._C = torch.zeros((nsims, 2, 2))
        self._Tbias = torch.zeros((nsims, 1, 2))
        self._b = torch.diag(torch.Tensor([1, 1])).repeat(nsims, 1, 1) * self.FRICTION

    def _Muact(self, q):
        qc, qp = q[:, :, 0].unsqueeze(1), q[:, :, 1].unsqueeze(1)
        self._M[:, 1, 0] = (self._Mp * self._L * torch.cos(qp)).squeeze()
        self._M[:, 1, 1] = (self._Mp * self._L ** 2).squeeze()
        return self._M[:, 1, :]

    def _Mact(self, q):
        qc, qp = q[:, :, 0].unsqueeze(1), q[:, :, 1].unsqueeze(1)
        self._M[:, 0, 0] = (self._Mp + self._Mc).squeeze()
        self._M[:, 0, 1] = (self._Mp * self._L * torch.cos(qp)).squeeze()
        return self._M[:, 0, :]

    def _Mfull(self, q):
        self._Mact(q)
        self._Muact(q)
        return self._M

    def _Cfull(self, x):
        qp, qdp = x[:, :, 1].unsqueeze(1), x[:, :, 3].unsqueeze(1)
        self._C[:, 0, 1] = (-self._Mp * self._L * qdp * torch.sin(qp)).squeeze()
        return self._C

    def _Tgrav(self, q):
        qc, qp = q[:, :, 0].unsqueeze(1), q[:, :, 1].unsqueeze(1)
        self._Tbias[:, 0, 1] = (-self._Mp * self.GRAVITY * self._L * torch.sin(qp)).squeeze()
        return self._Tbias

    def __call__(self, x, xd):
        q, qd = x[:, :, :2], x[:, :, 2:]
        M = self._Mfull(q)
        M_21, M_22 = M[:, 1, 0].unsqueeze(1), M[:, 1, 1].unsqueeze(1)
        Tp = (self._Tgrav(q) - (self._Cfull(x) @ qd.mT).mT)[:, :, 1]
        Tfric = (self._b @ qd.mT).mT[:, :, 1]
        qddc = xd[:, :, 2].clone()
        qddp = 1/M_22 * (Tp - Tfric - M_21 * qddc)
        xd = torch.cat((qd[:, :, 0], qd[:, :, 1], qddc, qddp), 1)
        return xd


if __name__ == "__main__":
    cp = Cartpole(1)
    x_init = torch.randn((1, 1, 4))
    xd_init = torch.randn((1, 1, 4))

    def test_func(x, xd):
        qc, qp, qd, qddc = x[:, :, 0], x[:, :, 1],  x[:, :, 2:], xd[:, :, 2]
        return torch.cat((qd[:, :, 0], qd[:, :, 1], qddc, -qddc * torch.cos(qp) - torch.sin(qp)), 1)

    ref = test_func(x_init, xd_init)
    res = cp(x_init, xd_init)
    print(torch.square(ref - res))

    x_init = torch.Tensor([0, 0.2, 0, 0]).view(1, 1, 4)
    xd_init = torch.Tensor([0, 0, 0, 0]).view(1, 1, 4)

    def integrate(func, x, xd, time, dt):
        xs = []

        for t in range(time):
            xd_new = func(x, xd)
            xd[:, :, :2] = xd_new[:, :2]
            x = x + xd_new * dt
            xs.append(x)

        return xs


    xs = integrate(cp, x_init, xd_init, 10000, 0.01)

    pos = [x[:, :, 1].item() for x in xs]
    print(pos)
    plt.plot(pos)
    plt.show()
