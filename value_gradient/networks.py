from mlp_torch import MLP
from net_utils_torch import LayerInfo
from utilities.torch_utils import gradify
from utilities.data_utils import *
import torch
import torch.nn.functional as F
from torch_device import device, is_cuda


class ValueFunction(MLP):
    def __init__(self, data_params: DataParams, layer_info: LayerInfo, apply_sigmoid=False):
        super(ValueFunction, self).__init__(layer_info, apply_sigmoid)
        self.loss = list()
        self._params = data_params
        self._v = torch.Tensor((self._params.n_batch, 1))
        self._dvdx = torch.Tensor(self._params.n_batch, self._params.n_state)
        self._dvdxx = torch.Tensor(self._params.n_batch, self._params.n_state, self._params.n_state)
        self._dvdx_desc = torch.Tensor(self._params.n_batch, self._params.n_state + self._params.n_desc)

    def dvdx(self, inputs):
        # Compute network derivative w.r.t state
        self._v = self.forward(inputs).requires_grad_()
        self._dvdx = torch.autograd.grad(
            self._v, inputs, grad_outputs=torch.ones_like(self._v), create_graph=True
        )[0][:, :self._params.n_state]
        return self._dvdx

    def dvdxx(self, inputs):
        self.dvdx(inputs)
        for i in range(self._params.n_batch):
            self._dvdxx[i] = torch.autograd.functional.hessian(
                self.forward, inputs[i]
            )[:self._params.n_state, :self._params.n_state]

        return self._dvdxx

    def dvdx_desc(self, inputs):
        self._v = self.forward(inputs).requires_grad_()
        self._dvdx_desc = torch.autograd.grad(
            self._v, inputs, grad_outputs=torch.ones_like(self._v), create_graph=True
        )[0]
        return self._dvdx_desc

    def get_dvdx(self, inputs):
        return self.dvdx(inputs).detach().clone()

    def get_dvdxx(self, inputs):
        return self.dvdxx(inputs).detach().clone()

    def update_grads_x(self, inputs):
        self._dvdx = self.dvdx(inputs).detach().clone()
        self._dvdxx = self.dvdxx(inputs).detach().clone()

    def update_grads(self, inputs):
        self._dvdx = self.dvdx(inputs).detach().clone()
        self._dvdxx = self.dvdxx(inputs).detach().clone()
        self._dvdx_desc = self.dvdx_desc(inputs).detach().clone()

    def update_dvdx(self, inputs):
        self._dvdx = self.dvdx(inputs).detach().clone()

    def update_dvdxx(self, inputs):
        self._dvdxx = self.dvdxx(inputs).detach().clone()

    def update_dvdx_desc(self, inputs):
        self._dvdx_desc = self.dvdx_desc(inputs).detach().clone()


class OptimalPolicy(torch.nn.Module):
    def __init__(
            self, policy: MLP, value: ValueFunction, params: DataParams,
            derivative=lambda x1, x2, dt=.01: (x2 - x1) / dt,
            integrate=lambda x1, x2, dt=0.01: x1 + x2 * dt
    ):
        super(OptimalPolicy, self).__init__()
        self._policy_net = policy
        self._v_net = value
        self._params = params
        self._derivative = derivative
        self._integrate = integrate
        self._final_pos = torch.zeros((self._params.n_batch, self._params.n_pos)).to(device)
        self._final_full_x = torch.zeros((self._params.n_batch, self._params.n_full_state)).to(device)
        self._final_x_desc = torch.zeros((self._params.n_batch, self._params.n_state + self._params.n_desc)).to(device)
        self._final_dxdt = torch.zeros((self._params.n_batch, self._params.n_vel * 2)).to(device)
        self._conds = torch.ones((self._params.n_batch, self._params.n_desc)).to(device)
        self._v = gradify(torch.zeros((self._params.n_batch, 1)), True)

    # Policy net need to output positions. From there derivatives can compute velocity.
    # Projection onto value function computes new acceleration, then integrate to get state
    def forward(self, inputs):
        inputs.requires_grad_()
        pos, vel = inputs[:, 0:self._params.n_pos], inputs[:, self._params.n_pos:self._params.n_vel + self._params.n_pos]
        self._final_pos = self._policy_net(inputs)
        self._v = self._v_net(inputs)
        self._final_dxdt[:, :self._params.n_vel] = self._derivative(pos, self._final_pos)
        self._final_dxdt[:, self._params.n_vel:] = self._derivative(
            vel, self._final_dxdt[:, :self._params.n_vel]
        )

        # Compute network derivative w.r.t state
        dvdx = self._v_net.dvdx(inputs)

        # Return the new state form
        xd = self._final_dxdt - dvdx * (F.relu(
            (dvdx * self._final_dxdt).sum(dim=1)
        ) / (dvdx ** 2).sum(dim=1))[:, None]

        pos = self._integrate(
            pos, self._final_full_x[:, self._params.n_pos:self._params.n_state]
        )

        self._final_x_desc = torch.column_stack(
            (self._final_full_x[:, :self._params.n_state], inputs[:, self._params.n_state:])
        )

        return self._final_full_x, self._final_x_desc