import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.autograd.functional import jacobian
# import matplotlib.pyplot as plt
from models import Cartpole, ModelParams
from animations.cartpole import animate_cartpole, init_fig_cp

fig_3, p, r, width, height = init_fig_cp(0)

# PMP implementation
dt, T, nx, nu, tol, delta, discount = 0.01, 300, 4, 1, 1e-8, .002, 1
A = torch.Tensor(([1, dt], [0, 1])).requires_grad_()
B = torch.Tensor(([0, 1])).requires_grad_()
R = (torch.Tensor(([0.0001])))
Qr = torch.diag(torch.Tensor([1, 1, 1*dt, 1*dt])) * 5
Qf = torch.diag(torch.Tensor([1, 1, 1*dt, 1*dt])) * 100
Q = torch.diag(torch.Tensor([1, 1, 1*dt, 1*dt])) * 1

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
mult = torch.zeros((T+1, 1, nx))


def state_encoder(x: torch.Tensor):
    b, r, c = x.shape
    x = x.reshape((b, r * c))
    qc, qp, v = x[:, 0].clone().unsqueeze(1), x[:, 1].clone().unsqueeze(1), x[:, 2:].clone()
    qp = torch.pi ** 2 * torch.sin(qp/2)
    return torch.cat((qc, qp, v), 1).reshape((b, r, c))


def f(x: torch.Tensor, u: torch.Tensor):
    return (A@x.T).T + B * u.T


def l(x: torch.Tensor, u: torch.Tensor, t):
    x, u = x.unsqueeze(0), u.unsqueeze(0)
    return (state_encoder(x) @ Qr @ state_encoder(x).mT + u @ R @ u.mT) * discount ** t.item()


def h(x: torch.Tensor, t):
    return (state_encoder(x.reshape(1, 1, nx)) @ Qf @ state_encoder(x.reshape(1, 1, nx)).mT) * discount**t.item()


def dldx(x: torch.Tensor, u: torch.Tensor, t):
    return jacobian(l, (x, u, t))[0].squeeze().reshape(1, nx)


def dldu(x: torch.Tensor, u: torch.Tensor, t):
    return jacobian(l, (x, u, t))[1].squeeze().reshape(1, nu)


def dhdx(x: torch.Tensor, t):
    return jacobian(h, (x, t))[0].squeeze().reshape(1, nx)


def dfdx(x: torch.Tensor, u: torch.Tensor):
    return jacobian(f, (x, u))[0].squeeze().reshape(nx, nx)


def dfdu(x: torch.Tensor, u: torch.Tensor):
    return jacobian(f, (x, u))[1].squeeze().reshape(nx, nu)


def forward(x: torch.Tensor, us: torch.Tensor):
    xs[0] = x
    for t in range(T):
        t_tensor = torch.Tensor([t])
        x_next, x, u = xs[t+1].clone().requires_grad_(), xs[t].clone().requires_grad_(), us[t].clone().requires_grad_()
        xs[t+1], fx[t], fu[t] = f(x, u), dfdx(x, u), dfdu(x, u)
        ls[t], lx[t], lu[t] = l(x, u, t_tensor), dldx(x, u, t_tensor), dldu(x, u, t_tensor)

    return xs.clone()


def backward(mult: torch.Tensor, lx: torch.Tensor, fx: torch.Tensor, xs: torch.Tensor,
             Hu: torch.Tensor, lu: torch.Tensor, fu: torch.Tensor):
    mult[-1] = dhdx(xs[-1], torch.Tensor([T-1]))
    for t in range(T-1, -1, -1):
        mult[t] = lx[t] + mult[t+1] @ fx[t]
        Hu[t] = lu[t] + mult[t+1] @ fu[t]


def optimize(us: torch.Tensor):
    us = us - (delta * Hu)
    return us, (delta * Hu)


def PMP(x: torch.Tensor, us: torch.Tensor, max_iter=1):
    error, xs, iter = 1e10, None, 0
    print(f"init {x}")
    while error >= tol and iter < max_iter:
        xs = forward(x, us)
        backward(mult, lx, fx, xs, Hu, lu, fu)
        us, steps = optimize(us)
        error = torch.mean(steps.norm(dim=2))
        print(f"Adaption: {error}, StateT: {xs[-1]}")
        if iter % 10 == 0:
        #     # plt.plot(xs[:, :, 0].clone().detach().numpy())
            cart = xs[:, 0, 0].cpu().detach().numpy()
            pole = xs[:, 0, 1].cpu().detach().numpy()
            animate_cartpole(cart, pole, fig_3, p, r, width, height, skip=2)
            plt.pause(0.001)

        iter += 1
    return xs, us


def pmp_MPC(x: torch.Tensor, us: torch.Tensor):
    xs, us = PMP(x, us, max_iter=1)
    u = us[0, :, :]
    us = us.roll(-1, 0)
    us[-1, :, :] *= 0
    return u


if __name__ == "__main__":
    cp_params = ModelParams(2, 2, 1, 4, 4)
    cp = Cartpole(1, cp_params, 'cpu', mode='pfl')
    cp_anim = Cartpole(1, cp_params, 'cpu', mode='pfl')


    def f(x, u):
        x, u = x.unsqueeze(0), u.unsqueeze(0)
        xd = cp(x, u)
        x_next = x + xd * dt
        return x_next.reshape(1, cp_params.nx)

    def l(x, u, t):
        x, u = x.unsqueeze(0), u.unsqueeze(0)
        q = x[:, :, :cp_params.nq].clone()
        T = cp.inverse_dynamics(x, u)
        u_loss = T @ torch.linalg.inv(cp._Mfull(q)) @ T.mT
        return state_encoder(x) @ Qr @ state_encoder(x).mT + u_loss * 1/5

    x = torch.Tensor([0, torch.pi, 0, 0]).reshape(1, 1, nx)
    us = torch.rand((T, 1, 1)) * 2
    cart, pole = [0, 0], [0, 0]
    poles = []

    # for i in range(1000):
    #     cart[0], pole[0] = x[:, :, 0].item(), x[:, :, 1].item()
    #     u = pmp_MPC(x, us).reshape(1, 1, nu)
    #     xd = cp_anim(x.reshape(1, 1, nx), u)
    #     x = x + xd * 0.01
    #
    #     if torch.abs(state_encoder(x)[:, :, 1]).item() < 0.012:
    #         print("STABALIZING")
    #         Qr = Qf
    #     else:
    #         Qr = Q
    #
    #     print(f"x: {x}, u: {u}")
    #     cart[1], pole[1] = x[:, :, 0].item(), x[:, :, 1].item()
    #
    #     # plt.scatter(i, pole[1])
    #     animate_cartpole(np.array(cart), np.array(pole), fig_3, p, r, width, height)

    xs, us = PMP(x, us, max_iter=300)

    thetas = np.linspace(0, 6 * np.pi, 300)
    enc = lambda x: np.pi**2 * np.sin(x/2)
    enc_2 = lambda x: np.cos(x) - 1
    theta_enc = enc(thetas)**2
    plt.plot(thetas, theta_enc)
    plt.show()
