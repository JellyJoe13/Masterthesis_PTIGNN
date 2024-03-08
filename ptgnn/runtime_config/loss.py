import torch.nn.functional


def graphgym_cross_entropy_loss(
        pred,
        true,
        reduction='mean',
        **kwargs
):
    """
    Adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/graphgym/loss.html
    """
    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if pred.ndim > 1 and true.ndim == 1:
        # multiclass but true label is not one hot encoded but argmax
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        return torch.nn.functional.nll_loss(pred, true), pred

    else:
        # binary or multilabel
        # make true label a float (could be an int category)
        true = true.float()

        # init loss class (not permanent necessary)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction=reduction)

        return loss_function(pred, true), torch.sigmoid(pred)


def l1_loss(pred, true):
    loss = torch.nn.L1Loss()
    return loss(pred, true), pred