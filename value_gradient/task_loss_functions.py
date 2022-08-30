import torch

_di_x_gain = None


def set_gains__(running_gain, batch_size):
    global _di_x_gain
    rc = len(running_gain)
    _di_x_gain = torch.diag(torch.tensor(running_gain))
    _di_x_gain = _di_x_gain.reshape((1, rc, rc))
    _di_x_gain = _di_x_gain.repeat(batch_size, 1, 1)


def task_loss(x: torch.Tensor):
    global _di_x_gain
    b, r, c = x.size()[0], x.size()[1], x.size()[2]
    xg = torch.bmm(x, _di_x_gain)
    return torch.sum(torch.bmm(x, xg.view(b, c, r)))

