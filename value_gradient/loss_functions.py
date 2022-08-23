import torch.nn.functional as F
from torch.autograd import Function
from utilities.mj_utils import MjBatchOps
from utilities.torch_utils import *
from networks import ValueFunction


_value_net_ = None
_batch_op_ = None

dtype = torch.float
device = torch.device("cpu")


def set_batch_ops__(b_ops: MjBatchOps):
    global _batch_op_
    _batch_op_ = b_ops


def set_value_net__(value_net: ValueFunction):
    global _value_net_
    _value_net_ = value_net


def value_dt_loss(v_curr, v_next, dt, value_func):
    return (F.relu(1 + (value_func(v_curr) - value_func(v_next)) / dt)).mean()


def value_goal_loss(goal, value_func):
    return 10 * value_func(goal)


class value_lie_loss(Function):
    @staticmethod
    def forward(ctx, x_desc, u):
        ctx.constant = u
        x_cpu, u_cpu = x_desc[:, :_batch_op_.params.n_state].cpu(), u.cpu()
        x_np, u_np = tensor_to_np(x_desc[:, :_batch_op_.params.n_state]), tensor_to_np(u)
        dvdx_np = tensor_to_np(_value_net_._dvdx)
        dvdxx_np = tensor_to_np(_value_net_._dvdxx)
        gu_np = _batch_op_.b_gu(u_np).reshape((_batch_op_.params.n_batch, _batch_op_.params.n_state))
        dxdt_np = _batch_op_.b_dxdt(x_np, u_np)
        ctx.save_for_backward(
            x_desc, np_to_tensor(x_cpu), np_to_tensor(gu_np), np_to_tensor(dxdt_np), np_to_tensor(dvdx_np), np_to_tensor(dvdxx_np)
        )
        return torch.Tensor(
            dvdx_np[:, :_batch_op_.params.n_state].dot(gu_np.T) + dvdx_np[:, :_batch_op_.params.n_state].dot(dxdt_np.T)
        )

    @staticmethod
    def backward(ctx, grad_output):
        x_desc, x_cpu, gu, dxdt, dvdx, dvdxx = ctx.saved_tensors
        dvdxx = dvdxx.reshape(
            _batch_op_.params.n_batch, _batch_op_.params.n_state + _batch_op_.params.n_desc, _batch_op_.params.n_state + _batch_op_.params.n_desc
        )
        dfdx = _batch_op_.b_dfdx(x_cpu)
        return grad_output * torch.Tensor(
            dvdxx * gu + dvdxx * dxdt + dvdx * dfdx
        ), None


class ctrl_effort_loss(Function):
    @staticmethod
    def forward(ctx, x_full):
        x_full_cpu = x_full.cpu()
        x_full_np = tensor_to_np(x_full)
        qfrcs = _batch_op_.b_qfrcs(x_full_np)
        ctx.save_for_backward(x_full_cpu, np_to_tensor(qfrcs))
        loss = torch.sum(torch.square(torch.tensor(qfrcs, device=device, dtype=dtype)))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_full, qfrcs = ctx.saved_tensors
        frcs_np = tensor_to_np(qfrcs)
        x_full_np = tensor_to_np(x_full)
        dfinvdx = _batch_op_.b_dfinvdx_full(x_full_np)
        grad_1 = grad_output * 2 * frcs_np * dfinvdx.reshape(
            _batch_op_.params.n_batch, _batch_op_.params.n_vel, _batch_op_.params.n_full_state
        )
        return grad_1


class ctrl_clone_loss(Function):
    @staticmethod
    def forward(ctx, x_full, u_star):
        ctx.constant = u_star
        x_full_cpu = x_full.cpu()
        u_star_cpu = u_star.cpu()
        x_full_np = tensor_to_np(x_full)
        qfrcs = _batch_op_.b_qfrcs(x_full_np)
        ctx.save_for_backward(x_full_cpu, np_to_tensor(qfrcs), u_star_cpu)
        return torch.sum(torch.square(u_star - qfrcs))

    @staticmethod
    def backward(ctx, grad_output):
        x_full, qfrc, u_star = ctx.saved_tensors
        x_full_np = x_full.detach().numpy().astype('float64')
        dfinvdx = _batch_op_.b_dfinvdx_full(x_full_np)
        grad_1 = grad_output * 2 * (qfrc - u_star) * dfinvdx
        return grad_1, None


