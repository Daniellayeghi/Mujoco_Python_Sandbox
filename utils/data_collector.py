from collections import namedtuple

__orderd_ids = [b"p", b"v", b"s", b"i", b"t"]
__attr_names = ["pos_list", "vel_list", "state_list", "ctrl_list", "time_list"]
__id_name_map = dict(zip(__attr_names, __orderd_ids))
__attr_vals = [[], [], [], [], []]
__DataCollectorMap = namedtuple("DataCollector", __attr_names)
__DataCollectorMap(*__attr_vals)


def get_data_collector():
    return __DataCollectorMap(*__attr_vals)


def update_data_collector(dc: namedtuple, buffer: dict):
    for name, value in dc._asdict().items():
        if buffer['id'][0] == __id_name_map[name]:
            value.append(buffer['val'])
