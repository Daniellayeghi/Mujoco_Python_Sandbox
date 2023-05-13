import torch


def extract_value_net(model):
    return model.policy.mlp_extractor.value_net, model.policy.value_net

def generate_value_net(policy_net: str, env, policy_type):
    model = policy_type(policy_net, env)
    return extract_value_net(model)

class ValueNet(torch.nn.Module):
    def __init__(self, value_feature, value_head):
        super(ValueNet, self).__init__()
        self._vfeat = value_feature
        self._vhead = value_head


    def forward(self, x):
        x_hat = self._vfeat(x)
        v = self._vhead(x_hat)
        return v
