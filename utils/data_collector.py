import numpy as np
from collections import namedtuple

orderd_ids = [b"p", b"v", b"s", b"i", b"t"]
attr_names = ["pos_list", "vel_list", "state_list", "ctrl_list", "time_list"]
id_name_map = dict(zip(attr_names, orderd_ids))
attr_vals = [[], [], [], [], []]
DataCollectorMap = namedtuple("DataCollector", attr_names)
DataCollectorMap(*attr_vals)


def get_data_collector():
    return DataCollectorMap(*attr_vals)


def update_data_collector(dc: namedtuple, buffer: dict):
    for name, value in dc._asdict().items():
        if buffer['id'][0] == id_name_map[name]:
            value.append(buffer['val'])
