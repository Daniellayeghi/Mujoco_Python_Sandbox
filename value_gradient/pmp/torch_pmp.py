import torch
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt

# PMP implementation
dt, T, nx, nu, tol, delta = 0.01, 20, 2, 1, 1e-6, 0.05
A = torch.Tensor(([1, dt], [0, 1])).requires_grad_()
B = torch.Tensor(([0, 1])).requires_grad_()
R = (torch.Tensor(([0.0001])))
Qr = (torch.Tensor(([1, 0], [0, .1*dt]))) * 2
Qf = (torch.Tensor(([10, 0], [0, 10*dt]))) * 2

# decisions
us = torch.rand((T, 1, nu))
xs = torch.zeros((T+1, 1, nx))

# losses
ls = torch.zeros(T, 1, 1)
lx = torch.zeros((T, 1, nx))
lu = torch.zeros((T, 1, nu))
Hu = torch.zeros((T, 1, nu))

# dynamics
fx = torch.zeros((T + 1, nx, nx))
fu = torch.zeros((T, nx, nu))
mult = torch.zeros((T, 1, nx))


def f(x: torch.Tensor, u: torch.Tensor):
    return (A@x.T).T + B * u.T


def l(x: torch.Tensor, u: torch.Tensor):
    return x @ Qr @ x.mT + u @ R @ u.mT


def h(x: torch.Tensor):
    return x @ Qf @ x.mT


def dldx(x: torch.Tensor, u: torch.Tensor):
    return jacobian(l, (x, u))[0].squeeze().reshape(1, nx)


def dldu(x: torch.Tensor, u: torch.Tensor):
    return jacobian(l, (x, u))[1].squeeze().reshape(1, nu)


def dhdx(x: torch.Tensor):
    return jacobian(h, (x)).squeeze().reshape(1, nx)


def dfdx(x: torch.Tensor, u: torch.Tensor):
    return jacobian(f, (x, u))[0].squeeze().reshape(nx, nx)


def dfdu(x: torch.Tensor, u: torch.Tensor):
    return jacobian(f, (x, u))[1].squeeze().reshape(nx, nu)


def forward(x: torch.Tensor, us: torch.Tensor):
    xs[0] = x
    for t in range(T):
        x_next, x, u = xs[t+1].clone().requires_grad_(), xs[t].clone().requires_grad_(), us[t].clone().requires_grad_()
        xs[t+1], fx[t], fu[t] = f(x, u), dfdx(x, u), dfdu(x, u)
        ls[t], lx[t], lu[t] = l(x, u), dldx(x, u), dldu(x, u)

    return xs.clone()


def backward(mult: torch.Tensor, lx: torch.Tensor, fx: torch.Tensor, xs: torch.Tensor,
             Hu: torch.Tensor, lu: torch.Tensor, fu: torch.Tensor):
    mult[-1] = dhdx(xs[-1])
    for t in range(T-1, 0, -1):
        mult[t - 1] = lx[t - 1] + mult[t] @ fx[t - 1]
        Hu[t-1] = lu[t-1] + mult[t] @ fu[t-1]


def optimize(us: torch.Tensor):
    us = us - (delta * Hu)
    return us, (delta * Hu)


def PMP(x: torch.Tensor, us: torch.Tensor):
    error, xs, iter = 1e10, None, 1
    while error >= tol or iter < 300:
        xs = forward(x, us)
        backward(mult, lx, fx, xs, Hu, lu, fu)
        us, steps = optimize(us)
        error = torch.mean(steps.norm(dim=2))
        print(f"Adaption: {error}, StateT: {xs[-1]}")
        if iter % 30 == 0:
            plt.plot(xs[:, :, 0].clone().detach().numpy())
            plt.pause(0.001)

        iter += 1
    return xs, us


if __name__ == "__main__":
    x = torch.Tensor([1, 0]).reshape(1, 1, nx)
    us = torch.zeros((T, 1, 1))
    xs, us = PMP(x, us)
