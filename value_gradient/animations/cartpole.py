import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import numpy as np
import time

# Create Animation:
# Setup Figure:


def animate_cartpole(x_vec, theta_vec, fig, p, r, width, height, dt=0.01, skip = 1):
    x_pole = x_vec + 1 * np.sin(theta_vec)
    y_pole = np.cos(theta_vec)
    sim_time = x_vec.shape[0] * dt# Length of Simulation
    t_vec = np.arange(0, sim_time, dt)
    # Initialize State Vectors:
    vec_size = len(t_vec)

    plt.ion()
    plt.show()

    for i in range(0, vec_size, skip):
        # Update Pendulum Arm:
        x_pole_arm = [x_vec[i], x_pole[i]]
        y_pole_arm = [0, y_pole[i]]
        p.set_data(x_pole_arm, y_pole_arm)
        # Update Cart Patch:
        r.set(xy=(x_vec[i] - width / 2,  - height / 2))
        # Update Drawing:
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(.00001)


def animate_double_cartpole(x_vec, theta1_vec, theta2_vec, fig, p, r, width, height, dt=0.01, skip=1):
    x_pole_1 = x_vec - np.sin(theta1_vec)
    x_pole_2 = x_pole_1 - np.sin(theta2_vec)
    y_pole_1 = np.cos(theta1_vec)
    y_pole_2 = y_pole_1 + np.cos(theta2_vec)
    p1, p2 = p[0], p[1]

    sim_time = x_vec.shape[0] * dt# Length of Simulation
    t_vec = np.arange(0, sim_time, dt)
    # Initialize State Vectors:
    vec_size = len(t_vec)

    plt.ion()
    plt.show()

    for i in range(0, vec_size, skip):
        # Update Pendulum Arm:
        x_pole1_arm = [x_vec[i], x_pole_1[i]]
        y_pole1_arm = [0, y_pole_1[i]]
        x_pole2_arm = [x_pole_1[i], x_pole_2[i]]
        y_pole2_arm = [y_pole_1[i], y_pole_2[i]]
        p1.set_data(x_pole1_arm, y_pole1_arm)
        p2.set_data(x_pole2_arm, y_pole2_arm)
        # Update Cart Patch:
        r.set(xy=(x_vec[i] - width / 2,  - height / 2))
        # Update Drawing:
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(.00001)


def init_fig_cp(x_init):
    fig_anim, ax_anim = plt.subplots(num=4)
    p, = ax_anim.plot([], [], color='royalblue')
    min_lim = -10
    max_lim = 10
    ax_anim.axis('equal')
    ax_anim.set_xlim([min_lim, max_lim])
    ax_anim.set_ylim([-4, 4])
    ax_anim.set_xlabel('X')
    ax_anim.set_ylabel('Y')
    ax_anim.set_title('Cartpole Simulation:')
    title = "simulation"

    # Setup Animation Writer:
    # FPS = 20
    # sample_rate = int(1 / (dt * FPS))  # Real Time Playback
    # dpi = 300
    # writerObj = FFMpegWriter(fps=FPS)

    # Initialize Patch: (Cart)
    width = 1  # Width of Cart
    height = width / 2  # Height of Cart
    xy_cart = (x_init - width / 2, - height / 2)  # Bottom Left Corner of Cart
    r = Rectangle(xy_cart, width, height, color='cornflowerblue')  # Rectangle Patch
    ax_anim.add_patch(r)  # Add Patch to Plot

    # Draw the Ground:
    ground = ax_anim.hlines(-height / 2, min_lim, max_lim, colors='black')
    height_hatch = 0.25
    width_hatch = max_lim - min_lim
    xy_hatch = (min_lim, - height / 2 - height_hatch)
    ground_hatch = Rectangle(xy_hatch, width_hatch, height_hatch, facecolor='None', linestyle='None', hatch='/')
    ax_anim.add_patch(ground_hatch)
    return fig_anim, p, r, width, height


def init_fig_dcp(x_init):
    fig_anim, ax_anim = plt.subplots(num=4)
    p1, = ax_anim.plot([], [], color='royalblue')
    p2, = ax_anim.plot([], [], color='red')
    min_lim = -10
    max_lim = 10
    ax_anim.axis('equal')
    ax_anim.set_xlim([min_lim, max_lim])
    ax_anim.set_ylim([-4, 4])
    ax_anim.set_xlabel('X')
    ax_anim.set_ylabel('Y')
    ax_anim.set_title('Cartpole Simulation:')
    title = "simulation"

    # Setup Animation Writer:
    # FPS = 20
    # sample_rate = int(1 / (dt * FPS))  # Real Time Playback
    # dpi = 300
    # writerObj = FFMpegWriter(fps=FPS)

    # Initialize Patch: (Cart)
    width = 1  # Width of Cart
    height = width / 2  # Height of Cart
    xy_cart = (x_init - width / 2, - height / 2)  # Bottom Left Corner of Cart
    r = Rectangle(xy_cart, width, height, color='cornflowerblue')  # Rectangle Patch
    ax_anim.add_patch(r)  # Add Patch to Plot

    # Draw the Ground:
    ground = ax_anim.hlines(-height / 2, min_lim, max_lim, colors='black')
    height_hatch = 0.25
    width_hatch = max_lim - min_lim
    xy_hatch = (min_lim, - height / 2 - height_hatch)
    ground_hatch = Rectangle(xy_hatch, width_hatch, height_hatch, facecolor='None', linestyle='None', hatch='/')
    ax_anim.add_patch(ground_hatch)
    return fig_anim, (p1, p2), r, width, height