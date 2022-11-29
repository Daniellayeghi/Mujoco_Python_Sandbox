import math
import mujoco
import torch.linalg

from neural_value_synthesis_diffeq import *
from utilities.torch_utils import save_models
from utilities.mujoco_torch import torch_mj_set_attributes, SimulationParams, torch_mj_inv


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path("/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator_sparse.xml")
    d = mujoco.MjData(m)
    sim_params = SimulationParams(3 * 2, 2 * 2, 1 * 2, 1 * 2, 1 * 2, 2, 1, 201)
    prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, -1, 10000, 2
    Q = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    Qb = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    R = torch.diag(torch.Tensor([0.5])).repeat(sim_params.nsim, 1, 1).to(device)
    Qf = torch.diag(torch.Tensor([0, 1, 0, 1])).repeat(sim_params.nsim, 1, 1).to(device)
    torch_mj_set_attributes(m, sim_params)

    def loss_func(x: torch.Tensor):
        nsim, r, c = x.shape
        x_dist = x[:, :, :sim_params.nq].view(nsim, r, sim_params.nq).clone()
        x_task = x.view(x.shape).clone()
        loss_dist = torch.linalg.vector_norm(x_dist, dim=2).view(nsim, r, 1)
        loss_task = (x_task @ Q @ x_task.mT)
        return torch.div(loss_task, loss_dist) + loss_dist

    def batch_state_loss(x: torch.Tensor):
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t-1, nsim, r, c).clone()
        x_final = x[-1, :, :, :].view(1, nsim, r, c).clone()
        l_task = x_run @ Q @ x_run.mT
        l_dist = torch.linalg.vector_norm(x_run, dim=3).view(t-1, nsim, r, 1)
        l_running = torch.sum(torch.div(l_task, l_dist) + l_dist, dim=0).squeeze()
        l_terminal = (x_final @ Qf @ x_final.mT).squeeze()
        return torch.mean(l_running + l_terminal)

    def batch_ctrl_loss(xxd: torch.Tensor):
        return torch.mean(torch_mj_inv.apply(xxd))

    def batch_state_ctrl_loss(x, xxd):
        return batch_state_loss(x) + 1 * batch_ctrl_loss(xxd)

    # Initialise the network
    S_init = torch.FloatTensor(sim_params.nqv, sim_params.nqv).uniform_(0, 5).to(device)
    lin_value_func = LinValueFunction(sim_params.nqv, S_init).to(device)
    nn_value_func = NNValueFunction(sim_params.nqv).to(device)
    dyn_system = DynamicalSystem(nn_value_func, loss_func, sim_params).to(device)
    optimizer = torch.optim.AdamW(dyn_system.parameters(), lr=3e-2)

    # Assemble initial conditions
    q_init_u_act = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0.5, 1)
    q_init_act = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(1.5, 3)
    q_init = torch.cat((q_init_u_act, q_init_act), 2)
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nq).uniform_(-1, 1) * 5
    x_init = torch.cat((q_init, qd_init), 2).to(device)

    # Assemble function grid
    pos_arr = torch.linspace(-10, 10, 100).to(device)
    vel_arr = torch.linspace(-10, 10, 100).to(device)
    f_mat = torch.zeros((100, 100)).to(device)
    [X, Y] = torch.meshgrid(pos_arr.squeeze().cpu(), vel_arr.squeeze().cpu())
    time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)

    while diff > tol and iteration < max_iter:
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

        # if True:
        #     with torch.no_grad():
        #         plot_2d_funcition(pos_arr, vel_arr, [X, Y], f_mat, lin_value_func, trace=traj, contour=True)

    save_models("./neural_value", lin_value_func)
    plt.show()
