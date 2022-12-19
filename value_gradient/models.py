import torch


class Cartpole:
    LENGTH = 1
    MASS_C = 1
    MASS_P = 1
    GRAVITY = -9.81
    FRICTION = 0.1

    def __init__(self, nsims):
        self._L = torch.ones((nsims,  1, 1)) * self.LENGTH
        self._Mp = torch.ones((nsims,  1, 1)) * self.MASS_P
        self._Mc = torch.ones((nsims,  1, 1)) * self.MASS_C
        self._Ma = torch.zeros((nsims, 1, 2))
        self._Mu = torch.zeros((nsims, 1, 2))
        self._M = torch.zeros((nsims, 2, 2))
        self._C = torch.zeros((nsims, 2, 2))
        self._Tbias = torch.zeros((nsims, 1, 2))
        self._b = torch.diag(torch.Tensor([1, 1])).repeat(nsims, 1, 1) * self.FRICTION

    def _Muact(self, q):
        qc, qp = q[:, :, 0], q[:, :, 1]
        self._M[:, 1, 0] = self._Mp * self._L * torch.cos(qp)
        self._M[:, 1, 1] = self._Mp * self._L ** 2
        return self._M[:, 1, :]

    def _Mact(self, q):
        qc, qp = q[:, :, 0], q[:, :, 1]
        self._M[:, 0, 0] = self._Mp + self._Mc
        self._M[:, 0, 1] = self._Mp * self._L * torch.cos(qp)
        return self._M[:, 0, :]

    def _Mfull(self, q):
        self._Mact(q)
        self._Muact(q)
        return self._M

    def _Cfull(self, x):
        qp, qdp = x[:, :, 1], x[:, :, 3]
        self._C[:, 0, 1] = -self._Mp * self._L * qdp * torch.sin(qp)
        return self._C

    def _Tgrav(self, q):
        qc, qp = q[:, :, 0], q[:, :, 1]
        self._Tbias[:, 0, 1] = -self._Mp * self.GRAVITY * self._L * torch.sin(qp)
        return self._Tbias

    def __call__(self, x, xd):
        q, qd = x[:, :, :2], x[:, :, 2:]
        M = self._Mfull(q)
        M_21, M_22 = M[:, 1, 0], M[:, 1, 1]
        Tp = (self._Tgrav(q) - (self._Cfull(x) @ qd.mT).mT)[:, :, 1]
        Tfric = (self._b @ qd.mT).mT[:, :, 1]
        qddc = xd[:, :, 2].clone()
        qddp = 1/M_22 * (Tp - Tfric - M_21 @ qddc.mT)
        xd = torch.Tensor([qddc, qddp])
        return xd


if __name__ == "__main__":
    cp = Cartpole(1)
    x_init = torch.zeros((1, 1, 4))
    xd_init = torch.zeros((1, 1, 4))
    x_init[:, :, 1] = 0.2
    xd_init[:, :, 2] = 1
    print(f"{cp(x_init, xd_init)}")
