
import torch.nn.functional as F
from torch.autograd import Function
from utilities.mj_utils import MjBatchOps
from utilities.torch_utils import *
from task_loss_functions import task_loss
from networks import ValueFunction


_value_net_ = None
_batch_op_ = None
_dt_ = None

dtype = torch.float


def set_batch_ops__(b_ops: MjBatchOps):
    global _batch_op_
    _batch_op_ = b_ops


def set_value_net__(value_net: ValueFunction):
    global _value_net_
    _value_net_ = value_net


def set_dt_(dt: float):
    global _dt_
    _dt_ = dt


def value_goal_loss(goal):
    return 10 * _value_net_(goal)


def value_dt_loss_auto(x_next, x_curr):
    real_loss = 1 + (_value_net_(x_curr) - _value_net_(x_next)) / _dt_
    positive_loss = F.relu(real_loss)

    return positive_loss


class value_descent_loss(Function):
    @staticmethod
    def forward(ctx, x, u):
        dfdt = _batch_op_.b_dxdt(x, u).view(_batch_op_.params.n_batch, _batch_op_.params.n_vel * 2, 1)
        loss = torch.sum(_value_net_._dvdx * dfdt)
        ctx.save_for_backward(x, u, _value_net_._dvdx, _value_net_._dvdxx, dfdt)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, u, dvdx, dvdxx, dfdt = ctx.saved_tensors
        dfdx = _batch_op_.b_dfdx_u(x, u).view(_batch_op_.params.n_batch, _batch_op_.params.n_state, _batch_op_.params.n_state)
        dfdu = _batch_op_.b_dfdu(x, u).view(_batch_op_.params.n_batch, _batch_op_.params.n_state, _batch_op_.params.n_ctrl)
        vtdx = torch.sum(dvdxx * dfdt) + torch.sum(dvdx * dfdx)
        vtdu = torch.sum(dvdx * dfdu)
        return grad_output * vtdx, grad_output, vtdu


class value_dt_loss(Function):
    @staticmethod
    def forward(ctx, x_next, x_curr):
        ctx.constant = x_curr
        real_loss = 1 + (_value_net_(x_curr) - _value_net_(x_next)) /_dt_
        dvdx_desc = _value_net_._dvdx_desc
        ctx.save_for_backward(x_next, real_loss, dvdx_desc)
        positive_loss = F.relu(real_loss)

        return positive_loss

    @staticmethod
    def backward(ctx, grad_output):
        x_next, real_loss, dvdx_desc = ctx.saved_tensors
        dldv = 1/_dt_ * _value_net_(x_next).view((_batch_op_.params.n_batch, 1, 1))
        drelu_dloss = (1 * (real_loss > 0)).view((_batch_op_.params.n_batch, 1, 1))

        grad_1 = (drelu_dloss * torch.bmm(
            dvdx_desc.view((_batch_op_.params.n_batch, _batch_op_.params.n_state + _batch_op_.params.n_desc, 1)), dldv
        )).view((_batch_op_.params.n_batch, _batch_op_.params.n_state + _batch_op_.params.n_desc))

        grad_1.to(grad_output.device)
        return grad_output * grad_1, None


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

        real_loss = torch.bmm(
                dvdx.view(_batch_op_.params.n_batch, 1, _batch_op_.params.n_state),
                gu.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1) +
                dxdt.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1)
            )

        positive_loss = F.relu(real_loss).view(_batch_op_.params.n_batch, 1)

        ctx.save_for_backward(
            x_cpu, gu, dxdt, dvdx, dvdxx, real_loss
        )

        return positive_loss.to(x_desc.device)

    @staticmethod
    def backward(ctx, grad_output):
        x_cpu, gu, dxdt, dvdx, dvdxx, real_loss = ctx.saved_tensors
        dfdx = _batch_op_.b_dfdx(x_cpu)
        dlie_dx = torch.zeros(
            (_batch_op_.params.n_batch, _batch_op_.params.n_state + _batch_op_.params.n_desc)
        ).to(grad_output.device)
        drelu_dx = 1 * (real_loss > 0).view((_batch_op_.params.n_batch, 1))

        dlie_dx[:, :_batch_op_.params.n_state] = (drelu_dx * (torch.bmm(
            dvdxx, (gu + dxdt).view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1)
        ) + torch.bmm(
            dfdx.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, _batch_op_.params.n_state),
            dvdx.view(_batch_op_.params.n_batch, _batch_op_.params.n_state, 1)
        )).view((_batch_op_.params.n_batch, _batch_op_.params.n_state))).to(grad_output.device)

        return grad_output * dlie_dx, None


class ctrl_effort_loss(Function):
    @staticmethod
    def forward(ctx, x_full):
        x_full_cpu = x_full.cpu()
        x_full_np = tensor_to_np(x_full)
        qfrcs = np_to_tensor(_batch_op_.b_qfrcs(x_full_np))
        ctx.save_for_backward(x_full_cpu, qfrcs)
        loss = torch.sum(torch.square(qfrcs), 1).to(x_full.device)

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
        grad_1 = (2 * grad_1.view(_batch_op_.params.n_batch, _batch_op_.params.n_full_state)).to(grad_output.device)

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
        loss = torch.sum(torch.square(u_star_cpu - qfrcs), 1).to(x_full.device)
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
        grad_1 = (2 * grad_1.view(_batch_op_.params.n_batch, _batch_op_.params.n_full_state)).to(grad_output.device)

        return grad_output.view(_batch_op_.params.n_batch, 1) * grad_1, None



