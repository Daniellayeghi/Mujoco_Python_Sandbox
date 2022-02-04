import struct
from math import floor
import zmq
import numpy as np


class SystemDim:
    CTRL_SIZE = 9
    POS_SIZE = 11
    VEL_SIZE = 11
    STATE_SIZE = POS_SIZE + VEL_SIZE


class MessageParser:
    CTRL_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': b'c', 'NUM': SystemDim.CTRL_SIZE}
    POS_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': b'p', 'NUM': SystemDim.POS_SIZE}
    VEL_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': b'v', 'NUM': SystemDim.VEL_SIZE}
    STATE_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': b's', 'NUM': SystemDim.STATE_SIZE}

    TYPE_msg = [CTRL_msg, POS_msg, VEL_msg, STATE_msg]

    def __init__(self):
        pass

    def parse_msg(self, msg):
        msg_size = len(msg)
        msg_id = struct.unpack('c', msg[0:1])
        types = [msg_type for msg_type in self.TYPE_msg if msg_id[0] == msg_type['id']]
        elem_num = floor(msg_size / types[0]["Size"])
        return struct.unpack(types[0]['Type'] * elem_num, msg[1:msg_size]), msg_id

    def deep_parser(self, msg):
        id_idx = 0
        data_idx = id_idx + 1
        while id_idx < len(msg):
            msg_id = struct.unpack('c', msg[id_idx:data_idx])
            types = [msg_type for msg_type in self.TYPE_msg if msg_id[0] == msg_type['id']]
            data_end = data_idx + (types[0]['Size'] * types[0]['NUM'])
            parsed_msg = struct.unpack(types[0]['Type'] * types[0]['NUM'], msg[data_idx:data_end])
            # print(msg_id, parsed_msg)
            id_idx = data_end
            data_idx = id_idx + 1


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.bind("tcp://*:5555")
    while True:
        socket.send(b'r')
        msg = socket.recv()
        print(msg)
        msg_p = MessageParser()
        msg_p.deep_parser(msg)
        #
        # message, identify = msg_p.parse_msg(msg)
        # ctrl_data = [np.array(message)]
        # print(message)