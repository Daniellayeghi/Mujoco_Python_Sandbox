import torch
import torch.utils.data as Data

class FVI:
    def __init__(self, net, loss_func, optimizer, batch_size=64, epochs=100):
        self._net = net
        self._floss = loss_func
        self._optimizer = optimizer
        self._bs = batch_size
        self._epoch = epochs
        self._x, self._target = None, None

    def _update_data(self, x, target, cat=True):
        state = self._x is not None and self._target is not None
        if state and cat:
            self._x = torch.cat((self._x, x), dim=0)
            self._target = torch.cat((self._target, target), dim=0)
        else:
            self._x, self._target = x.clone(), target.clone()

        return self._x, self._target

    def _assemble_loader(self, x, target, cat=True):
        target = target.reshape(target.shape[0]*target.shape[1], 1)
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[-1])
        self._x, self._target = self._update_data(x, target, cat=cat)
        _ds = Data.TensorDataset(self._x.squeeze(), self._target)

        _loader = Data.DataLoader(
            dataset=_ds,
            batch_size=self._bs,
            shuffle=True
        )

        return _ds, _loader

    def train(self, x, target, cat=True):

        with torch.set_grad_enabled(True):
            _ds, _loader = self._assemble_loader(x, target, cat=cat)

            for epoch in range(self._epoch):
                for step, (inputs, target) in enumerate(_loader):
                    self._optimizer.zero_grad()
                    # Compute net prediction and loss:
                    pred = self._net(0, inputs)
                    loss = self._floss(pred.squeeze(), target.squeeze())
                    # Backprop through the net
                    loss.backward()
                    self._optimizer.step()

                print(f"Loss: {loss}, epoch {epoch}")