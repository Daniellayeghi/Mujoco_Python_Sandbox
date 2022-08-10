import torch
import torch.nn.functional as F
from torch.autograd import Function
from utilities.mj_utils import batch_f, batch_f_inv2


def value_dt_loss(v_curr, v_next, dt, value_func):

    return (F.relu(1 + (value_func(v_curr) - value_func(v_next)) / dt)).mean()


def value_goal_loss(goal, value_func):
    return 10 * value_func(goal)


class value_lie_loss(Function):
    @staticmethod
    def forward(ctx, x, u, value_func, data, data_cp, model):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class ctrl_effort_loss(Function):
    @staticmethod
    def forward(ctx, x, frc, data, data_cp, model, params):
        batch_f_inv2(frc, x, data, data_cp, model, params)
        ctx.save_for_backward(frc)
        torch.square_(frc)

    @staticmethod
    def backward(ctx, grad_output):
        pass


class ctrl_clone_loss(Function):
    @staticmethod
    def forward(ctx, x, u_new, u_star, data, data_cp, model, params):
        batch_f_inv2(u_new, x, data, data_cp, model, params)
        ctx.save_for_backward(x, u_star)
        return torch.square_(u_star - u_new)

    @staticmethod
    def backward(ctx, grad_output):
        x, u_star = ctx.saved_tensors
        return grad_output


