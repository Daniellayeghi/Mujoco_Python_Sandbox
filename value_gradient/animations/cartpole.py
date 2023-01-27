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
    # fig, ax = plt.subplots()
    # p, = ax.plot([], [], color='royalblue')
    # min_lim = -5
    # max_lim = 5
    # ax.axis('equal')
    # ax.set_xlim([min_lim, max_lim])
    # ax.set_ylim([min_lim, max_lim])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('Cartpole Simulation:')
    # title = "simulation"
    #
    # # Setup Animation Writer:
    # # FPS = 20
    # # sample_rate = int(1 / (dt * FPS))  # Real Time Playback
    # # dpi = 300
    # # writerObj = FFMpegWriter(fps=FPS)
    #
    # # Initialize Patch: (Cart)
    # width = 1  # Width of Cart
    # height = width / 2  # Height of Cart
    # xy_cart = (x_vec[0] - width / 2, - height / 2)  # Bottom Left Corner of Cart
    # r = Rectangle(xy_cart, width, height, color='cornflowerblue')  # Rectangle Patch
    # ax.add_patch(r)  # Add Patch to Plot
    #
    # # Draw the Ground:
    # ground = ax.hlines(-height / 2, min_lim, max_lim, colors='black')
    # height_hatch = 0.25
    # width_hatch = max_lim - min_lim
    # xy_hatch = (min_lim, - height / 2 - height_hatch)
    # ground_hatch = Rectangle(xy_hatch, width_hatch, height_hatch, facecolor='None', linestyle='None', hatch='/')
    # ax.add_patch(ground_hatch)

    # Animate:
    # with writerObj.saving(fig, title + ".mp4", dpi):
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
            # Save Frame:
            # writerObj.grab_frame()


def init_fig(x_init):
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