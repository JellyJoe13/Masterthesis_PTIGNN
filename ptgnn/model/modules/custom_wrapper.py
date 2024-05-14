import torch


class CustomWrapper(torch.nn.Module):
    """
    Wrapper around a torch module that puts the return value of a forward into the batch x node embedding.
    """
    def __init__(
            self,
            layer_to_wrap: torch.nn.Module
    ):
        """
        Init of CustomWrapper class.

        :param layer_to_wrap: layer to wrap
        :type layer_to_wrap: torch.nn.Module
        """
        super(CustomWrapper, self).__init__()
        self.layer_to_wrap = layer_to_wrap

    def forward(self, batch):
        """
        (Wrapped) Forward, output of original layer is written to ``batch.x`` and batch is returned.

        :param batch: batch input object
        :return: batch object with updated node features
        """
        x = self.layer_to_wrap(batch)
        batch.x = x
        return batch


class InputOutputWrapper(torch.nn.Module):
    """
    Wrapper around a torch module that puts the return value of a forward into the batch x node embedding and uses
    normal network internally as batch.x is passed to wrapped module.
    """
    def __init__(
            self,
            layer_to_wrap: torch.nn.Module
    ):
        """
        Init of CustomWrapper class.

        :param layer_to_wrap: layer to wrap
        :type layer_to_wrap: torch.nn.Module
        """
        super(InputOutputWrapper, self).__init__()
        self.layer_to_wrap = layer_to_wrap

    def forward(self, batch):
        """
        (Wrapped) Forward, output of original layer is written to ``batch.x`` and batch is returned.

        :param batch: batch input object
        :return: batch object with updated node features
        """
        x = self.layer_to_wrap(batch.x)
        batch.x = x
        return batch
