import torch
import torch.nn.functional as F
from torch.autograd import Function
from utilities.mj_utils import MjBatchOps
from utilities.torch_utils import *
from mujoco.derivative import MjDerivative
import mujoco
from collections import namedtuple

_value_net_ = None
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
    def forward(ctx, x, u):
        ctx.constant = u
        x_full_cpu, u_cpu = x.cpu(), u.cpu()
        x_full_np = x.detach().numpy().astype('float64')
        u_np = u.detach().numpy().astype('float64')
        dvdx_np = _value_net_.dvdx(x).detach().numpy().astype('float64')


    @staticmethod
    def backward(ctx, grad_output):
        pass


class ctrl_effort_loss(Function):
    @staticmethod
    def forward(ctx, x_full):
        x_full_cpu = x_full.cpu()
        x_full_np = x_full.detach().numpy().astype('float64')
        qfrcs = _batch_op_.b_qfrcs(x_full_np)
        ctx.save_for_backward(x_full_cpu, np_to_tensor(qfrcs))
        loss = torch.square(torch.tensor(qfrcs, device=device, dtype=dtype))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_full, qfrcs = ctx.saved_tensors
        frcs_np = qfrcs.detach().numpy().astype('float64')
        x_full_np = x_full.detach().numpy().astype('float64')
        dfinvdx = _batch_op_.b_dfinvdx_full(x_full_np)
        grad_1 = grad_output * 2 * frcs_np * dfinvdx.reshape(
            _batch_op_.params.n_vel, _batch_op_.params.n_full_state
        )
        return grad_1


class ctrl_clone_loss(Function):
    @staticmethod
    def forward(ctx, x_full, u_star):
        ctx.constant = u_star
        x_full_cpu = x_full.cpu()
        u_star_cpu = u_star.cpu()
        x_full_np = x_full.detach().numpy().astype('float64')
        qfrcs = _batch_op_.b_qfrcs(x_full_np)
        ctx.save_for_backward(x_full_cpu, np_to_tensor(qfrcs), u_star_cpu)
        return torch.square(u_star - qfrcs)

    @staticmethod
    def backward(ctx, grad_output):
        x_full, qfrc, u_star, = ctx.saved_tensors
        x_full_np = x_full.detach().numpy().astype('float64')
        dfinvdx = _batch_op_.b_dfinvdx_full(x_full_np)
        grad_1 = grad_output * 2 * (qfrc - u_star) * dfinvdx
        return grad_1, None


