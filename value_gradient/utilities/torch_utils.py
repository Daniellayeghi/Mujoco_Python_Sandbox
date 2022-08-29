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
    x_torch = torch.Tensor(x)
    if cuda:
        x_torch = x_torch.cuda()
    if grad:
        x_torch = x_torch.requires_grad_()
    return x_torch


def to_cuda(x: torch.Tensor, grad=False):
    return x.cuda().requires_grad_() if grad else x.cuda()


def tensor_to_np(x: torch.Tensor):
    return x.detach().cpu().numpy().astype('float') if x.is_cuda else x.detach().numpy().astype('float')
