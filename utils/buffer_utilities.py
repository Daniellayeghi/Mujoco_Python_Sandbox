import struct
from math import floor
import zmq
import numpy as np


class MessageParser:
    DDP_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': b'i'}
    PI_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': b'p'}
    STATE_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': b's'}
    TYPE_msg = [DDP_msg, PI_msg, STATE_msg]

    def __init__(self):
        pass

    def parse_msg(self, msg):
        msg_size = len(msg)
        msg_id = struct.unpack('c', msg[msg_size-1: msg_size])
        types = [msg_type for msg_type in self.TYPE_msg if msg_id[0] == msg_type['id']]
        elem_num = floor(msg_size / types[0]["Size"])
        return struct.unpack(types[0]['Type'] * elem_num, msg[0:msg_size - 1]), msg_id


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")
    while True:
        msg = socket.recv()
        msg_p = MessageParser()
        message, identify = msg_p.parse_msg(msg)
        ctrl_data = [np.array(message)]

