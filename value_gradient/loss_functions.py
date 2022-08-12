import torch
import torch.nn.functional as F
from torch.autograd import Function
from utilities.mj_utils import MjBatchOps
from mujoco.derivative import MjDerivative

__batch_op = None


def __set_derivative_operator__(op: MjDerivative):
    global __batch_op
    __batch_op = op


def value_dt_loss(v_curr, v_next, dt, value_func):

    return (F.relu(1 + (value_func(v_curr) - value_func(v_next)) / dt)).mean()


def value_goal_loss(goal, value_func):
    return 10 * value_func(goal)


class value_lie_loss(Function):
    @staticmethod
    def forward(ctx, x, u, value_func):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class ctrl_effort_loss(Function):
    @staticmethod
    def forward(ctx, x_full, frc, dfinv_dx, op: MjBatchOps):
        op.b_finv_full_x(frc, x_full)
        ctx.save_for_backward(frc, x_full, dfinv_dx, op)
        return torch.square_(frc)

    @staticmethod
    def backward(ctx, grad_output):
        frcs, x_full, dfinv_dx, op = ctx.saved_tensors
        op.b_dfinvdx_full(dfinv_dx, x_full)
        return grad_output * 2 * frcs * dfinv_dx.reshape(op.params.n_vel, op.params.n_full_state)


class ctrl_clone_loss(Function):
    @staticmethod
    def forward(ctx, x_full, frc, u_star, dfinv_dx, op):
        op.b_dfinvdx_full(frc, x_full)
        ctx.save_for_backward(x_full, u_star)
        ctx.save_for_backward(frc, u_star, x_full, dfinv_dx, op)
        return torch.square_(u_star - frc)

    @staticmethod
    def backward(ctx, grad_output):
        frc, u_star, x_full, dfinv_dx, op = ctx.saved_tensors
        op.b_dfinvdx_full(dfinv_dx, x_full)
        return grad_output * 2*(frc - u_star)*dfinv_dx


