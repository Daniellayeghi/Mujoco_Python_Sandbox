from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import zmq
from collections import namedtuple
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
from time import sleep
import struct
from collections import deque
import threading

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")
State = namedtuple('State', 'time qpos qvel act udd_state')

ByteParams = {"NumCtrl": 3, "NumCheck": 1, "CtrlSize": 8, "CheckSize": 1}
ctrl_ddp = [0 for _ in range(ByteParams["NumCtrl"])]
ctrl_pi = [0 for _ in range(ByteParams["NumCtrl"])]
ctrl_comb = [0 for _ in range(ByteParams["NumCtrl"] * 2)]
ctrl_names = ["DDP", "PI"]
joints = [(i+1 + (i+1)%2)/2 for i in range(ByteParams["NumCtrl"]*2)]


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
        # print("Received request: %s" % bytearray(message))
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
    nSamples = 1600
    curves = []
    for idx in range(nPlots):
        name = f"{ctrl_names[idx % (ByteParams['NumCtrl'] - 1)]} Joint {joints[idx]}"
        curve = pg.PlotCurveItem(pen=(idx, nPlots * 1.3), name=name)
        plot.addItem(curve)
        curve.setPos(0, idx * 2)
        curves.append(curve)

    plot.setYRange(-2, 2)
    plot.setXRange(0, nSamples)
    plot.resize(600, 900)

    ptr = 0
    lastTime = time()
    fps = None
    count = 0
    list_d = [deque(np.random.random(nSamples), maxlen=nSamples) for _ in range(nPlots)]

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
