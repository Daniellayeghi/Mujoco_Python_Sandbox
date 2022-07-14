from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import pandas as pd

from collections import namedtuple
from time import sleep
import argparse

State = namedtuple('State', 'time qpos qvel act udd_state')
if __name__ == "__main__":

    # Add the arguments
    my_parser = argparse.ArgumentParser(description='List the content of a folder')
    my_parser.add_argument(
        'ctrl_file', metavar='path', type=str, nargs='?', default="~/Repos/OptimisationBasedControl/data/ctrl_files_di.csv"
    )
    my_parser.add_argument(
        'state_file', metavar='path', type=str, nargs='?', default="~/Repos/OptimisationBasedControl/data/state_files_di.csv"
    )
    args = my_parser.parse_args()
    ctrl_data = pd.read_csv(args.ctrl_file, float_precision='round_trip', header=None).to_numpy()
    state_data = pd.read_csv(args.state_file, float_precision='round_trip', header=None).to_numpy()

    model = load_model_from_path(
        "/home/daniel/Repos/OptimisationBasedControl/models/doubleintegrator.xml"
    )
    sim = MjSim(model)
    viewer = MjViewer(sim)

    for idx, x in enumerate(state_data):
        init = State(time=0, qpos=x[0], qvel=0, act=0, udd_state={})
        sim.set_state(init)
        for u in ctrl_data[:, idx]:
            sim.data.ctrl[0] = u
            sim.step()
            viewer.render()
            sleep(0.01)
            print(f"{(x[1] - sim.data.qpos[0])**2}")
