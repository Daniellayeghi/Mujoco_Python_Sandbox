import torch

_di_x_gain = None


def set_gains__(x_gain):
    global _di_x_gain
    _di_x_gain = x_gain


def task_loss(x: torch.Tensor):
    global _di_x_gain
    b, r, c = x.size()[0], x.size()[1], x.size()[2]
    xg = torch.bmm(x, _di_x_gain)
    return torch.sum(torch.bmm(x, xg.view(b, c, r)))

