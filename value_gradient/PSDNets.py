import torch
from torch import nn
import torch.nn.functional as F
from utilities.torch_device import device


class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=F.relu_, eps=0.01):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])
        self.act = activation
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, t, x):
        nsim = x.shape[0]
        time = torch.ones((nsim, 1, 1)).to(device) * t
        aug_x = torch.cat((x, time), dim=2)
        z = F.linear(aug_x, self.W[0], self.bias[0])
        z = self.act(z)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(aug_x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)

        return (F.linear(aug_x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]) + self.eps*(aug_x**2).sum(1)[:,None]


class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)


class MakePSD(nn.Module):
    def __init__(self, f, n, eps=0.01, d=1.0):
        super().__init__()
        self.f = f
        self.zero = torch.nn.Parameter(f(torch.zeros(1,n)), requires_grad=False)
        self.eps = eps
        self.d = d
        self.rehu = ReHU(self.d)

    def forward(self, t, x):
        x = x.reshape(x.shape[0], x.shape[-1])
        nsim = x.shape[0]
        time = torch.ones(nsim, 1) * t
        aug_x = torch.cat((x, time), dim=1)
        smoothed_output = self.rehu(self.f(aug_x) - self.zero)
        quadratic_under = self.eps*(aug_x**2).sum(1,keepdim=True)
        return (smoothed_output + quadratic_under).reshape(nsim, 1, 1)


class PosDefICNN(nn.Module):
    def __init__(self, layer_sizes, eps=0.1, negative_slope=0.05):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0]))
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.eps = eps
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)

    def forward(self, t, x):
        nsim, c = x.shape[0], x.shape[2]
        time = torch.ones((nsim, 1, 1)).to(device) * t
        aug_x = torch.cat((x, time), dim=2).reshape(nsim, c+1)
        z = F.linear(aug_x, self.W[0])
        F.softplus(z)

        for W,U in zip(self.W[1:-1], self.U[:-1]):
            z = F.linear(aug_x, W) + F.linear(z, F.softplus(U))*self.negative_slope
            z = F.softplus(z)

        z = F.linear(aug_x, self.W[-1]) + F.linear(z, F.softplus(self.U[-1]))
        return F.softplus(z) + self.eps*(aug_x**2).sum(1)[:,None]