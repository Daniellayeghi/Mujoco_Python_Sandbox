import torch
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
        dvdx = _value_net_._dvdx.cpu()
        dvdxx = _value_net_._dvdxx.cpu()
        gu = _batch_op_.b_gu(u_cpu).reshape((_batch_op_.params.n_batch, _batch_op_.params.n_state))
        dxdt = np_to_tensor(_batch_op_.b_dxdt(x_np, u_np))

        loss = F.relu(
            torch.bmm(
                dvdx.view(_batch_op_.params.n_batch, 1, _batch_op_.params.n_state),
                gu.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1) +
                dxdt.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1)
            )
        ).view(_batch_op_.params.n_batch, 1).to(x_desc.device)

        ctx.save_for_backward(
            x_desc, x_cpu, gu, dxdt, dvdx, dvdxx, loss
        )

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_desc, x_cpu, gu, dxdt, dvdx, dvdxx, relu_lie = ctx.saved_tensors
        dfdx = _batch_op_.b_dfdx(x_cpu)

        print(1 * (relu_lie > 0))
        dlie_dx = torch.ones((_batch_op_.params.n_batch, _batch_op_.params.n_state + _batch_op_.params.n_desc))
        dlie_dx[:, :_batch_op_.params.n_state] = (torch.bmm(
            dvdxx, (gu + dxdt).view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1)
        ) + torch.bmm(
            dfdx.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, _batch_op_.params.n_state),
            dvdx.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1)
        )).view((_batch_op_.params.n_batch, _batch_op_.params.n_state))


        return grad_output * dlie_dx.to(x_desc.device), None


class ctrl_effort_loss(Function):
    @staticmethod
    def forward(ctx, x_full):
        x_full_cpu = x_full.cpu()
        x_full_np = tensor_to_np(x_full)
        qfrcs = _batch_op_.b_qfrcs(x_full_np)
        ctx.save_for_backward(x_full_cpu, np_to_tensor(qfrcs))
        loss = torch.sum(torch.square(torch.tensor(qfrcs, device=device, dtype=dtype)), 1).to(x_full.device)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_full, qfrcs = ctx.saved_tensors
        # Any inplace error might require copy/cloning x_full
        dfinvdx = _batch_op_.b_dfinvdx_full(x_full)
        grad_1 = torch.bmm(
            qfrcs.view(_batch_op_.params.n_batch, 1, _batch_op_.params.n_vel),
            dfinvdx.view(_batch_op_.params.n_batch, _batch_op_.params.n_vel, _batch_op_.params.n_full_state)
        )
        grad_1 = (2 * grad_1.view(_batch_op_.params.n_batch, _batch_op_.params.n_full_state)).to(x_full.device)
        return grad_output.view(_batch_op_.params.n_batch, 1) * grad_1


class ctrl_clone_loss(Function):
    @staticmethod
    def forward(ctx, x_full, u_star):
        ctx.constant = u_star
        x_full_cpu = x_full.cpu()
        u_star_cpu = u_star.cpu()
        x_full_np = tensor_to_np(x_full)
        qfrcs = np_to_tensor(_batch_op_.b_qfrcs(x_full_np))
        ctx.save_for_backward(x_full_cpu, qfrcs, u_star_cpu)
        loss = torch.sum(torch.square(u_star - qfrcs), 1).to(x_full.device)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x_full, qfrcs, u_star = ctx.saved_tensors
        # Any inplace error might require copy/cloning x_full
        dfinvdx = _batch_op_.b_dfinvdx_full(x_full)
        grad_1 = torch.bmm(
            (qfrcs - u_star).view(_batch_op_.params.n_batch, 1, _batch_op_.params.n_vel),
            dfinvdx.view(_batch_op_.params.n_batch, _batch_op_.params.n_vel, _batch_op_.params.n_full_state)
        )

        grad_1 = (2 * grad_1.view(_batch_op_.params.n_batch, _batch_op_.params.n_full_state)).to(x_full.device)

        return grad_output.view(_batch_op_.params.n_batch, 1) * grad_1, None


