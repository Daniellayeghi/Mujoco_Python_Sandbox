import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
from collections import namedtuple
device = 'cuda' if torch.cuda.is_available() else 'cpu'

__attr_names = ["layer_dims", "drop_id", "drop_rate", "act_funcs"]
LayerInfo = namedtuple("LayerInfo", __attr_names)


class InitNetwork(nn.Module):
    def __init__(self, layer_info: LayerInfo):
        super(InitNetwork, self).__init__()
        layers = layer_info.layer_dims
        modules = []

        for l_idx in range(len(layers)-1):
            if layer_info.act_funcs[l_idx] is None:
                args = [nn.Linear(layers[l_idx], layers[l_idx + 1])]
            else:
                args = [nn.Linear(layers[l_idx], layers[l_idx + 1]), layer_info.act_funcs[l_idx]]

            modules.extend(
                args
            )

        for l_idx in layer_info.drop_id:
            modules.insert(
                l_idx, nn.Dropout(layer_info.drop_rate)
            )

        self.nn = torch.nn.Sequential(*modules)

    def get_network(self):
        return self.nn