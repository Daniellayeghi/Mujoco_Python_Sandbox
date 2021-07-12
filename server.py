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

ctrl_ddp = [0]
ctrl_pi = [0]
ctrl_comb = [0, 0]


def update_plot():
    global ctrl, list_d, nSamples, curve, data, plot, lastTime, fps, nPlots
    lock.acquire()
    for i in range(nPlots):
        list_d[i].append(ctrl_comb[i])
        arr = np.array(list_d[i])
        curves[i].setData(arr.reshape(nSamples))
        now = time()
        dt = now - lastTime
        lastTime = now
    lock.release()


def update_mj(sim, socket, viewer=None):
    global ctrl_ddp, ctrl_pi, ctrl_comb
    prev_check = 10
    while True:
        message = socket.recv()
        # print("Received request: %s" % bytearray(message))
        lock.acquire()
        ctrl_ddp[0] = struct.unpack('d', bytearray(message)[0:8])[0]
        ctrl_pi[0] = struct.unpack('d', bytearray(message)[8:16])[0]
        check = struct.unpack('?', bytearray(message)[16:17])[0]
        ctrl_comb = [ctrl_ddp[0], ctrl_pi[0]]
        # print(f"{ctrl_comb}, {check}")
        lock.release()

        if prev_check != check and viewer is not None:
            prev_check = check
            for control in range(sim.data.ctrl.shape[0]):
                sim.data.ctrl[control] = ctrl_ddp[0]
                sim.step()
                viewer.render()


if __name__ == '__main__':
    lock = threading.Lock()
    model = load_model_from_path(
        "/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml"
        )

    sim = MjSim(model)
    viewer = MjViewer(sim)
    init = State(
        time=0,
        qpos=np.array([0, np.pi]),
        qvel=np.array([0, 0]),
        act=0,
        udd_state={}
    )

    sim.set_state(init)
    app = QtGui.QApplication([])
    plot = pg.plot()
    plot.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
    plot.setLabel('bottom', 'Index', units='B')

    nPlots = 2
    nSamples = 400
    curves = []
    for idx in range(nPlots):
        curve = pg.PlotCurveItem(pen=(idx, nPlots * 1.3))
        plot.addItem(curve)
        curve.setPos(0, 0)
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
    update_mj(sim, socket, viewer)
    # thread = threading.Thread(target=update_mj, args=(sim, socket))
    # thread.start()
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()
    # thread.join()
