import struct
from math import floor
import zmq
import numpy as np


class SystemDim:
    CTRL_SIZE = 9
    POS_SIZE = 11
    VEL_SIZE = 11
    STATE_SIZE = POS_SIZE + VEL_SIZE


message_ids = {"DDP": b'q', "PI": b'i', "POS": b'p', "VEL": b'v', "STATE": b's'}


class MessageParser:
    CTRL_PI_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': message_ids["PI"], 'NUM': SystemDim.CTRL_SIZE}
    CTRL_ILQR_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': message_ids["DDP"], 'NUM': SystemDim.CTRL_SIZE}
    POS_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': message_ids["POS"], 'NUM': SystemDim.POS_SIZE}
    VEL_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': message_ids["VEL"], 'NUM': SystemDim.VEL_SIZE}
    STATE_msg = {"Type": 'd', "Size": np.dtype('d').itemsize, 'id': message_ids["STATE"], 'NUM': SystemDim.STATE_SIZE}

    TYPE_msg = [CTRL_ILQR_msg, CTRL_PI_msg, POS_msg, VEL_msg, STATE_msg]

    def __init__(self):
        pass

    def deep_parser(self, msg, result_contain: list):
        id_idx = 0
        data_idx = id_idx + 1
        while id_idx < len(msg):
            msg_id = struct.unpack('c', msg[id_idx:data_idx])
            types = [msg_type for msg_type in self.TYPE_msg if msg_id[0] == msg_type['id']]
            data_end = data_idx + (types[0]['Size'] * types[0]['NUM'])
            parsed_msg = struct.unpack(types[0]['Type'] * types[0]['NUM'], msg[data_idx:data_end])
            result_contain.append({'id': msg_id, 'val': parsed_msg})
            id_idx = data_end
            data_idx = id_idx + 1


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")
    msg_p = MessageParser()
    result_contain = [{}]
    while True:
        # socket.send(b'r')
        msg = socket.recv()
        msg_p.deep_parser(msg, result_contain)
        print(result_contain)
