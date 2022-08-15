import torch
import torch.nn.functional as F
from torch.autograd import Function
from utilities.mj_utils import MjBatchOps
from mujoco.derivative import MjDerivative
import mujoco
from collections import namedtuple


_batch_op_ = None

dtype = torch.float
device = torch.device("cpu")


def set_batch_ops__(b_ops: MjBatchOps):
    global _batch_op_
    _batch_op_ = b_ops


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
    def forward(ctx, x_full, frc, dfinv_dx):
        ctx.constant = frc
        ctx.constant = dfinv_dx
        ctx.set_materialize_grads(False)
        x_full_cpu, frc_cpu, dfinv_dx_cpu = x_full.cpu(), frc.cpu(), dfinv_dx.cpu()
        ctx.save_for_backward(x_full_cpu, frc_cpu, dfinv_dx_cpu)
        frc_np = frc.detach().numpy().astype('float64')
        x_full_np = x_full.detach().numpy().astype('float64')
        _batch_op_.b_finv_x_full(frc_np, x_full_np)
        loss = torch.square(torch.tensor(frc_np, device=device, dtype=dtype))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_full, frcs, dfinv_dx = ctx.saved_tensors
        frcs_np = frcs.detach().numpy().astype('float64')
        x_full_np = x_full.detach().numpy().astype('float64')
        dfinv_dx_np = dfinv_dx.detach().numpy().astype('float64')
        _batch_op_.b_dfinvdx_full(dfinv_dx_np, x_full_np)
        grad_1, grad_2, grad_3 = None, None, None
        if ctx.needs_input_grad[0]:
            grad_1 = grad_output * 2 * frcs_np * dfinv_dx_np.reshape(
                _batch_op_.params.n_vel, _batch_op_.params.n_full_state
            )
        return grad_1, grad_2, grad_3


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
        return grad_output * 2 * (frc - u_star)*dfinv_dx


