import numpy as np
import torch
from torch.autograd import Variable


def to_variable(x, cuda=False):
    if isinstance(x, (tuple, list)):
        return tuple(to_variable(x) for x in x)
    else:
        x = Variable(x)
        if cuda:
            return x.cuda().requires_grad_()
        return x.requires_grad_()


def np_to_tensor(x: np.ndarray, cuda=False, grad=False):
    x_np = torch.Tensor(x)
    if cuda:
        x_np = x_np.cuda()
    if grad:
        x_np = x_np.requires_grad_()
    return x_np


def tensor_to_np(x: torch.Tensor):
    return x.detach().cpu().numpy().astype('float64') if x.is_cuda else x.detach().numpy().astype('float64')
