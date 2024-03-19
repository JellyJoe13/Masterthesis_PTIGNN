import torch


class CustomWrapper(torch.nn.Module):
    def __init__(
            self,
            layer_to_wrap: torch.nn.Module
    ):
        super(CustomWrapper, self).__init__()
        self.layer_to_wrap = layer_to_wrap

    def forward(self, batch):
        x = self.layer_to_wrap(batch)
        batch.x = x
        return batch
