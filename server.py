from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import zmq
from collections import namedtuple
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
import struct
from collections import deque
import threading
import argparse

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")
State = namedtuple('State', 'time qpos qvel act udd_state')

ctrl_names = ["DDP", "PI"]
ByteParams = {"NumCtrl": 3, "NumCheck": 1, "CtrlSize": 8, "CheckSize": 1}

def update_plot():

    global ctrl, list_d, nSamples, curve, data, plot, lastTime, nPlots, lock
    lock.acquire()
    for i in range(nPlots):
        list_d[i].append(ctrl_comb[i])
        arr = np.array(list_d[i])
        curves[i].setData(arr.reshape(nSamples))
        curves[i].name()
        now = time()
        dt = now - lastTime
        lastTime = now
    lock.release()


def update_mj(sim, socket, viewer=None):
    global ctrl_ddp, ctrl_pi, ctrl_comb, lock
    prev_check = 10
    while True:
        message_ilqr = socket.recv()
        message_pi = socket.recv()
        print("Received request: %s" % bytearray(message_ilqr))
        lock.acquire()
        for idx in range(ByteParams["NumCtrl"]):
            idx_1 = idx*ByteParams["CtrlSize"]
            idx_2 = idx_1 + ByteParams["CtrlSize"]
            ctrl_comb[idx*2] = struct.unpack('d', bytearray(message_ilqr)[idx_1:idx_2])[0]
            ctrl_comb[(idx*2)+1] = struct.unpack('d', bytearray(message_pi)[idx_1:idx_2])[0]
            # check = struct.unpack('?', bytearray(message_ilqr)[8:9])[0]
            # ctrl_comb = [ctrl_ddp, ctrl_pi]
        # print(f"{ctrl_ddp}, {check}")
        lock.release()

        # if prev_check != check and viewer is not None:
        #     prev_check = check
        #     for control in range(sim.data.ctrl.shape[0]):
        #         sim.data.ctrl[control] = ctrl_ddp[0]
        #         sim.step()
        #         viewer.render()


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('nctrl',
                       metavar='ctrl',
                       type=int)

    args = my_parser.parse_args()
    ByteParams["NumCtrl"] = 9 #args.nctrl

    ctrl_ddp = [0 for _ in range(ByteParams["NumCtrl"])]
    ctrl_pi = [0 for _ in range(ByteParams["NumCtrl"])]
    ctrl_comb = [0 for _ in range(ByteParams["NumCtrl"] * 2)]
    joints = [(i+1 + (i+1)%len(ctrl_names))/len(ctrl_names) for i in range(ByteParams["NumCtrl"]*2)]



    lock = threading.Lock()
    model = load_model_from_path(
        "/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml"
        )

    sim = MjSim(model)
    #viewer = MjViewer(sim)
    init = State(
        time=0,
        qpos=np.array([0, np.pi]),
        qvel=np.array([0, 0]),
        act=0,
        udd_state={}
    )

    sim.set_state(init)
    # app = QtGui.QApplication([])
    plot = pg.plot()
    plot.addLegend()
    plot.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
    plot.setLabel('bottom', 'Index', units='B')

    nPlots = ByteParams["NumCtrl"] * 2
    nSamples = 1200
    curves = []
    for idx in range(nPlots):
        name = f"{ctrl_names[idx % len(ctrl_names)]} Joint {(idx+1 + (idx+1)%len(ctrl_names))/len(ctrl_names)}"
        curve = pg.PlotCurveItem(pen=(idx, nPlots * 1.3), name=name)
        plot.addItem(curve)
        curve.setPos(0, idx * 10.1)
        curves.append(curve)

    plot.setYRange(-2, 2)
    plot.setXRange(0, nSamples)
    plot.resize(600, 900)

    ptr = 0
    lastTime = time()
    fps = None
    count = 0
    list_d = [deque(np.zeros(nSamples), maxlen=nSamples) for _ in range(nPlots)]

    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(0)

    import sys
    # update_mj(sim, socket, viewer)
    thread = threading.Thread(target=update_mj, args=(sim, socket))
    thread.start()
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    
    thread.join()
