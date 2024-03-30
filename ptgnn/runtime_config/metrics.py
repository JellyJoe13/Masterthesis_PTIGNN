# metric_best: accuracy, auc, ranking_accuracy_0.3, mae
# metric_agg: argmax, argmin

# metrics are sklearn metrics that are rounded, except auc, that is auroc (as it can be done on GPU)
# differentiation between modes
# task type differentiation
# task types: classification, classification_multilabel, regression_rank, regression
"""
The contents of this file are based on/adapted from:
https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/logger.py#L256
and
https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/metric_wrapper.py#L194
"""
import copy

import numpy as np
import pandas as pd
import torch
from scipy.stats import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, \
    mean_squared_error
from torchmetrics.functional import (
    accuracy,
    auroc,
    average_precision,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision,
    recall,
)

METRICS_DICT_TORCHMETRICS = {
    "accuracy": accuracy,
    "averageprecision": average_precision,
    "auroc": auroc,
    "confusionmatrix": confusion_matrix,
    "f1": f1_score,
    "fbeta": fbeta_score,
    "precisionrecallcurve": precision_recall_curve,
    "precision": precision,
    "recall": recall,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
}


def rounding_fn(score):
    return round(score, 2)


def classification_binary(
        true,
        pred,
        device: str,
        prefix: str
):
    # modify input
    true = torch.cat(true).squeeze(-1)
    true_int = true.int()
    pred = torch.cat(pred).squeeze(-1)
    pred_int = (pred >= 0.5).int()

    # auroc computation (AUC on GPU)
    if true.shape[0] < 1e7:
        # apparently computation for extremely large datasets is too slow
        auroc_score = auroc(
            pred.to(device),
            true_int.to(device),
            task='binary'
        ).item()
    else:
        auroc_score = 0.

    return {
        f'{prefix}_accuracy': rounding_fn(accuracy_score(true, pred_int)),
        f'{prefix}_precision': rounding_fn(precision_score(true, pred_int)),
        f'{prefix}_recall': rounding_fn(recall_score(true, pred_int)),
        f'{prefix}_f1': rounding_fn(f1_score(true, pred_int)),
        f'{prefix}_auc': rounding_fn(auroc_score),
    }


def classification_multilabel(
        true,
        pred,
        device: str,
        prefix: str
):

    # prepare data
    true, pred = torch.cat(true), torch.cat(pred)
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
    if true.ndim == 1:
        true = pred.unsqueeze(-1)

    # Send to GPU to speed up TorchMetrics if possible.
    true = true.to(device)
    pred = pred.to(device)

    # manage nan values
    # mode is by default set to 'ignore-mean-label'
    true_nans = torch.isnan(true)
    true = [
        true[..., ii][~true_nans[..., ii]] for ii in range(true.shape[-1])
    ]
    pred = [
        pred[..., ii][~true_nans[..., ii]] for ii in range(pred.shape[-1])
    ]

    def _compute_metric(
            pred,
            true,
            metric_name,
            **kwargs
    ):
        # fetch metric
        metric = METRICS_DICT_TORCHMETRICS[metric_name]

        # compute metric for each column and output nan if there is an error
        metric_val = []
        # cast to int is by default true
        for ii in range(len(true)):
            # threshold parameter lands in kwargs where it is not used, except in self.metric
            metric_val.append(
                metric(pred[ii], true[ii].int(), **kwargs)
            )

        return torch.nanmean(torch.stack(metric_val))

    # calculate metrics
    acc = _compute_metric(
        pred,
        true,
        "accuracy",
        threshold=0.
    )
    ap = _compute_metric(
        pred,
        true,
        "averageprecision",
        pos_label=1
    )
    auroc = _compute_metric(
        pred,
        true,
        "auroc",
        pos_label=1
    )

    return {
        f'{prefix}_accuracy': rounding_fn(acc),
        f'{prefix}_ap': rounding_fn(ap),
        f'{prefix}_auc': rounding_fn(auroc),
    }


def eval_spearmanr(
        true,
        pred
):
    """
    Compute Spearman Rho averaged across tasks.
    """
    res_list = []

    if true.ndim == 1:
        res_list.append(stats.spearmanr(true, pred)[0])

    else:
        for i in range(true.shape[1]):
            # ignore nan values
            is_labeled = ~np.isnan(true[:, i])
            res_list.append(
                stats.spearmanr(
                    true[is_labeled, i],
                    pred[is_labeled, i]
                )[0]
            )

    return {
        'spearmanr': sum(res_list) / len(res_list)
    }


def regression_fn(pred, true, prefix: str):
    true, pred = torch.cat(true), torch.cat(pred)

    return {
        f'{prefix}_mae': rounding_fn(mean_absolute_error(true, pred)),
        f'{prefix}_r2': rounding_fn(r2_score(true, pred, multioutput='uniform_average')),
        f'{prefix}_spearmanr': rounding_fn(
            eval_spearmanr(
                true.numpy(),
                pred.numpy()
            )['spearmanr']
        ),
        f'{prefix}_mse': rounding_fn(mean_squared_error(true, pred)),
        f'{prefix}_rmse': rounding_fn(mean_squared_error(true, pred, squared=False)),
    }


def get_ranking_accuracies(
        dataframe: pd.DataFrame,
        difference_threshold: float = 0.001
):
    """
    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L150
    """
    # assumes outputs and targets column are for pred and true
    smiles_groups_mean = dataframe.groupby(['ID', 'SMILES_nostereo'])[['targets', 'outputs']].mean().reset_index()

    stereoisomers_df = copy.deepcopy(smiles_groups_mean).rename(columns={'outputs': 'mean_predicted_score'})

    stereoisomers_df_margins = stereoisomers_df.merge(
        pd.DataFrame(stereoisomers_df.groupby('SMILES_nostereo').apply(lambda x: np.max(x.targets) - np.min(x.targets)),
                     columns=['difference']), on='SMILES_nostereo')
    top_1_margins = []
    margins = np.arange(0.1, 2.1, 0.1)  # originally it was ranging from 0.3 with no particular reason.

    for margin in margins:
        subset = stereoisomers_df_margins[
            np.round(stereoisomers_df_margins.difference, 1) >= np.round(margin, 1)]

        def _match(x):
            pred_scores = np.array(x.mean_predicted_score)
            if np.abs(pred_scores[0] - pred_scores[1]) < difference_threshold:
                return False
            return np.argmin(np.array(x.targets)) == np.argmin(pred_scores)

        top_1 = subset.groupby('SMILES_nostereo').apply(_match)

        if len(top_1) == 0:
            acc = np.nan
        else:
            acc = sum(top_1 / len(top_1))
        top_1_margins.append(acc)

    return margins, np.array(top_1_margins)


def regression_rank(
        pred,
        true,
        training_mode: str,  # train, val or test - don't do ranking for train
        dataframe: pd.DataFrame,
        prefix: str
):
    regression = regression_fn(pred, true, prefix=prefix)
    if training_mode == 'train':
        return regression
    true, pred = torch.cat(true), torch.cat(pred)

    dataframe["outputs"] = pred
    dataframe["targets"] = true
    margins, ranking_accuracy = \
        get_ranking_accuracies(dataframe)  # >= is the mode reported in ChIRo paper
    regression.update({f'{prefix}_ranking_accuracy_{m:.1f}': rounding_fn(a) for m, a in zip(margins, ranking_accuracy)})
    return regression


def metric_system(
        task_type: str,
        pred,
        true,
        device,
        training_mode: str = 'train',
        dataframe: pd.DataFrame = None,
        out_dim: int = 1,
        prefix: str = 'train'
):
    """
    Adapted from
    https://github.com/pyg-team/pytorch_geometric/blob/1675b019c7182dbdc4970561f0dbff6dec3ee299/torch_geometric/graphgym/logger.py#L226
    """
    if task_type == 'classification':
        if out_dim <= 2:
            return classification_binary(true=true, pred=pred, device=device, prefix=prefix)
        else:
            return classification_multilabel(true=true, pred=pred, device=device, prefix=prefix)

    elif task_type == "regression":
        return regression_fn(pred=pred, true=true, prefix=prefix)
    elif task_type == "regression_rank":
        return regression_rank(pred=pred, true=true, training_mode=training_mode, dataframe=dataframe, prefix=prefix)
    else:
        raise NotImplementedError("unknown task type")
