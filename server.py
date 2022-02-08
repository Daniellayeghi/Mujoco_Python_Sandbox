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
from utils.buffer_utilities import MessageParser, SystemDim, message_ids

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")
State = namedtuple('State', 'time qpos qvel act udd_state')

ctrl_names = ["DDP", "PI"]
ctrl_order = {"DDP": 0, "PI": 1}
ByteParams = {"NumCtrl": 3, "NumCheck": 1, "CtrlSize": 8, "CheckSize": 1}

ctrl_containers = [[]for _ in range (len(ctrl_names))]


def update_plot():
    global list_d, nSamples, curve, plot, lastTime, nPlots, lock
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


def parse_ctrl(socket, parser: MessageParser, result_container: list):
    while True:
        msg = socket.recv()
        parser.deep_parser(msg, result_container)
        print(result_container)
        with lock:
            for result in result_container:
                for ids in message_ids:
                    if ids == result['id']:
                        ctrl_containers[ctrl_order[ids]] = result['val']

            print(ctrl_containers)


def update_mj(socket):
    global ctrl_comb, lock
    while True:
        message_ilqr = socket.recv()
        message_pi = socket.recv()
        print("Received request: %s" % bytearray(message_ilqr))
        with lock:
            for idx in range(ByteParams["NumCtrl"]):
                idx_1 = idx*ByteParams["CtrlSize"]
                idx_2 = idx_1 + ByteParams["CtrlSize"]
                ctrl_comb[idx*2] = struct.unpack('d', bytearray(message_ilqr)[idx_1:idx_2])[0]
                ctrl_comb[(idx*2)+1] = struct.unpack('d', bytearray(message_pi)[idx_1:idx_2])[0]


if __name__ == '__main__':
    # my_parser = argparse.ArgumentParser(description='List the content of a folder')
    #
    # # Add the arguments
    # my_parser.add_argument('nctrl', metavar='ctrl', type=int)
    #
    # args = my_parser.parse_args()
    ByteParams["NumCtrl"] = 9 #args.nctrl

    ctrl_comb = [0 for _ in range(ByteParams["NumCtrl"] * 2)]

    lock = threading.Lock()

    # app = QtGui.QApplication([])
    plot = pg.plot()
    plot.addLegend()
    plot.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
    plot.setLabel('bottom', 'Index', units='B')

    nPlots = SystemDim.CTRL_SIZE * 2
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

    lastTime = time()
    fps = None
    count = 0
    list_d = [deque(np.zeros(nSamples), maxlen=nSamples) for _ in range(nPlots)]

    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(0)

    import sys
    msg_parse = MessageParser()
    result_contain = []
    thread = threading.Thread(target=parse_ctrl, args=(socket, msg_parse, result_contain))
    thread.start()
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    
    thread.join()
