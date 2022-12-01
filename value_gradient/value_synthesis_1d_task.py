import math
import mujoco
from neural_value_synthesis_diffeq import *
from utilities.torch_utils import save_models
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams, torch_mj_inv


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml")
    d = mujoco.MjData(m)
    sim_params = SimulationParams(3, 2, 1, 1, 1, 1, 50, 201)
    prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, 0.5, 100, 1.03
    Q = torch.diag(torch.Tensor([1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    R = torch.diag(torch.Tensor([0.5])).repeat(sim_params.nsim, 1, 1).to(device)
    Qf = torch.diag(torch.Tensor([1, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    torch_mj_set_attributes(m, sim_params)

    def loss_func(x: torch.Tensor):
        return x @ Q @ x.mT

    def batch_state_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t-1, nsim, r, c).clone()
        x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
        l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
        l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

        return torch.mean(l_running + l_terminal)

    def batch_ctrl_loss(xxd: torch.Tensor):
        return torch.mean(torch.sum(torch_mj_inv.apply(xxd), 0))

    def batch_state_ctrl_loss(x, xxd):
        return batch_state_loss(x) + 1 * batch_ctrl_loss(xxd)


    class LinValueFunction(nn.Module):
        """
        Value function is J = xSx
        """

        def __init__(self, n_in, Sinit):
            super(LinValueFunction, self).__init__()
            self.S = nn.Linear(2, 2, bias=-False)
            self.S.weight = nn.Parameter(Sinit)

        def forward(self, t: float, x: torch.Tensor):
            return self.S(x) @ x.mT


    class Encoder(nn.Module):
        def __init__(self, weight, sim_params: SimulationParams):
            super(Encoder, self).__init__()
            self.nin = sim_params.nqv
            self.E = nn.Linear(self.nin, 1).requires_grad_(False)
            self.E.weight = nn.Parameter(weight)

        def forward(self, x):
            return self.E(x)
    class NNValueFunction(nn.Module):
        def __init__(self, n_in):
            super(NNValueFunction, self).__init__()

            self.nn = nn.Sequential(
                nn.Linear(n_in, 4, bias=False),
                nn.Softplus(),
                nn.Linear(4, 1, bias=False)
            )

            def init_weights(net):
                if type(net) == nn.Linear:
                    torch.nn.init.xavier_uniform(net.weight)

            self.nn.apply(init_weights)

        def forward(self, t, x):
            return self.nn(x)

    S_init = torch.FloatTensor(sim_params.nqv, sim_params.nqv).uniform_(0, 5).to(device)
    # S_init = torch.Tensor([[1.7, 1], [1, 1.7]]).to(device)
    lin_value_func = LinValueFunction(sim_params.nqv, S_init).to(device)
    nn_value_func = NNValueFunction(sim_params.nqv).to(device)
    dyn_system = DynamicalSystem(nn_value_func, loss_func, sim_params).to(device)
    time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)
    optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-2)

    q_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 5
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 5
    # q_init = torch.ones((sim_params.nsim, 1, 1 * sim_params.nee))
    # qd_init = torch.zeros((sim_params.nsim, 1, 1 * sim_params.nee))
    x_init = torch.cat((q_init, qd_init), 2).to(device)
    pos_arr = torch.linspace(-10, 10, 100).to(device)
    vel_arr = torch.linspace(-10, 10, 100).to(device)
    f_mat = torch.zeros((100, 100)).to(device)
    [X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())

    while diff > tol or iteration > max_iter:
        optimizer.zero_grad()
        traj = odeint(dyn_system, x_init, time)
        acc = compose_acc(traj, 0.01)
        xxd = compose_xxd(traj, acc)
        loss = batch_state_ctrl_loss(traj, xxd)

        dyn_system.step *= step_size
        print(f"Stepping with {dyn_system.step}")

        diff = math.fabs(prev_cost - loss.item())
        prev_cost = loss.item()
        loss.backward()
        optimizer.step()

        for param in dyn_system.parameters():
            print(f"\n{param}\n")

        print(f"Epochs: {iteration}, Loss: {loss.item()}")
        iteration += 1

        if True:
            with torch.no_grad():
                plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, nn_value_func, trace=traj, contour=False)

    save_models("./neural_value", lin_value_func)
    plt.show()
