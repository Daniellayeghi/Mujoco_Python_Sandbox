import numpy as np
import zmq
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.ptime import time
from collections import deque
import threading
from utils.buffer_utilities import MessageParser, id_name_map

# Globals for the win
n_samples = 1200
list_d = [deque(np.zeros(n_samples), maxlen=n_samples) for _ in range(1)]
res_containers = [[] for _ in range(1)]
curves = []
ids = []
id_names = []
legend = dict()
total_plots = 0
first_run = True


def build_legends(result_container):
    ids = []
    names = []
    lens = []
    for result in result_container:
        for key in result:
            if key == "id":
                ids.append(result[key][0])
                names.append(id_name_map[result[key][0]])
            if key == "val":
                lens.append(len(result[key]))

    return ids, dict(zip(names, lens))


def update_plot():
    global list_d, n_samples, curve, plot, lastTime, total_plots, lock
    with lock:
        res_list = [list(res_containers[i]) for i in range(len(res_containers))]
        inter_res = [val for tup in zip(*res_list) for val in tup]
        for i in range(total_plots):
            list_d[i].append(inter_res[i])
            arr = np.array(list_d[i])
            curves[i].setData(arr.reshape(n_samples))
            curves[i].name()
            now = time()
            dt = now - lastTime
            lastTime = now


def parse_ctrl(socket, parser: MessageParser):
    global first_run, list_d, legend, total_plots, id_names, res_containers, ids
    while True:
        msg = socket.recv()
        parsed_msg = parser.parse(msg)

        if first_run:
            ids, legend = build_legends(parsed_msg)
            total_plots = sum(legend.values())
            id_names = list(legend.keys())
            res_containers = [[] for _ in range(len(id_names))]
            list_d = [deque(np.zeros(n_samples), maxlen=n_samples) for _ in range(total_plots)]
            first_run = not first_run

        with lock:
            for message in parsed_msg:
                for i, id in enumerate(ids):
                    if message['id'][0] == id:
                        res_containers[i] = message["val"]


if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")
    lock = threading.Lock()
    # app = QtGui.QApplication([])

    msg_parse = MessageParser()
    thread = threading.Thread(target=parse_ctrl, args=(socket, msg_parse))
    thread.start()

    while total_plots is 0:
        pass

    plot = pg.plot()
    plot.addLegend()
    plot.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
    plot.setLabel('bottom', 'Index', units='B')

    for idx in range(total_plots):
        name = f"{id_names[idx % len(id_names)]} Joint {(idx + 1 + (idx + 1) % len(id_names)) / len(id_names)}"
        curve = pg.PlotCurveItem(pen=(idx, total_plots * 1.3), name=name)
        plot.addItem(curve)
        curve.setPos(0, idx * 10.1)
        curves.append(curve)

    plot.setYRange(-2, 2)
    plot.setXRange(0, n_samples)
    plot.resize(600, 900)

    lastTime = time()
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(0)

    import sys
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    socket.close()
    context.destroy()
    thread.join()

