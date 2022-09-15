from net_utils_torch import LayerInfo, InitNetwork
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(nn.Module):
    def __init__(self, layer_info: LayerInfo, apply_sigmoid=False):
        super(MLP, self).__init__()
        self.mlp = InitNetwork(layer_info).get_network()

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)

        # Applying it to our net
        self.mlp.apply(init_weights)

        self.apply_sigmoid = apply_sigmoid

    def forward(self, input):
        output = self.mlp(input)
        if self.apply_sigmoid:
            probs = torch.sigmoid(output)
            return probs
        return output

