from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from time import sleep
import argparse

State = namedtuple('State', 'time qpos qvel act udd_state')
if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('path_csv',
                       metavar='path',
                       type=str)

    args = my_parser.parse_args()


    model = load_model_from_path(
        "/home/daniel/Repos/OptimisationBasedControl/models/assets_hand/hand/manipulate_pen.xml"
        )

    sim = MjSim(model)
    
    df = pd.read_csv(args.path_csv)

    ctrl_arr = df.to_numpy()
    ctrl_arr = np.delete(ctrl_arr, sim.data.ctrl.shape[0], 1)
    pos = np.zeros(25); pos[-1] = 1.57
    init = State(
        time=0,
        qpos=pos,
        qvel=np.zeros(25),
        act=0,
        udd_state={}
    )

    sim.set_state(init)
    viewer = MjViewer(sim)

    for time in range(len(ctrl_arr)):
        for control in range(sim.data.ctrl.shape[0]):
            sim.data.ctrl[control] = ctrl_arr[time][control]
        sim.step()
        viewer.render()
        sleep(0.01)
