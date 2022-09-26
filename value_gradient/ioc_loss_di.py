import torch
import torch.nn.functional as F
from torch.autograd import Function
from utilities.torch_utils import *
from utilities.data_utils import *
from task_loss_functions import task_loss
from networks import ValueFunction

_value_net_ = None

batch_size = 32
d_params = DataParams(3, 2, 1, 1, 1, 0, [1, 2], batch_size)
b_Rinv = torch.ones((batch_size, 1, 1))
b_Btran = torch.tensor([0.0005, 0.09999975]).repeat(d_params.n_batch, 1, 1)


def set_value_net__(value_net: ValueFunction):
    global _value_net_
    _value_net_ = value_net


class loss_value_proj(Function):
    @staticmethod
    def forward(ctx, x, u_star):
        ctx.constant = u_star
        b_B_R = 0.5 * torch.bmm(b_Rinv, b_Btran)
        b_proj_v = torch.bmm(b_B_R, _value_net_._dvdx.mT)
        b_loss = u_star.view(d_params.n_batch, d_params.n_ctrl, 1) + b_proj_v
        ctx.save_for_backward(x, _value_net_._dvdxx, b_B_R)
        return b_loss

    @staticmethod
    def backward(ctx, grad_output):
        x, dvdxx, b_B_R = ctx.saved_tensors
        grad_1 = torch.bmm(b_B_R, dvdxx.mT)
        return (grad_output * grad_1).view(batch_size, d_params.n_state), None