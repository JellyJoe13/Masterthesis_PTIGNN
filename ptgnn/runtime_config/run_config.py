import typing
from collections import defaultdict
from functools import partial

import pandas as pd
import torch.cuda
import torch_geometric
from tqdm import tqdm

from ptgnn.dataset import DATASET_DICT
from ptgnn.loading.custom_assembly import custom_loader
from ptgnn.loading.load import UniversalLoader
from ptgnn.loading.subsetting import subset_dataset
from ptgnn.model.framework.custom_model import CustomModel
from ptgnn.optimizing import OPTIMIZER_DICT, SCHEDULER_DICT
from ptgnn.runtime_config.config import priority_merge_config, optional_fetch
from ptgnn.runtime_config.loss import l1_loss, graphgym_cross_entropy_loss
from ptgnn.runtime_config.metrics import metric_system

from ray import train


def fetch_loaders(
        data_config: typing.Dict[str, typing.Any],
        verbose: bool = True
) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Function used to build the loaders based on the parameters passed in the config dict object.

    :param data_config: config object storing the parameters required for the creation of the loaders
    :type data_config: typing.Dict[str, typing.Any]
    :param verbose: Whether or not progress should be plotted to output
    :type verbose: bool
    :return: train, validation and test loaders
    :rtype: typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    dataset_config = data_config['dataset']
    # load dataset
    ds_type = DATASET_DICT.get(dataset_config['type'])
    train_ds = ds_type(**dataset_config, split='train')
    test_ds = ds_type(**dataset_config, split="test")
    val_ds = ds_type(**dataset_config, split="val")

    # subset data
    if 'subset_size' in data_config:
        train_ds = subset_dataset(train_ds, subset_size=data_config['subset_size'])
        test_ds = subset_dataset(test_ds, subset_size=data_config['subset_size'])
        val_ds = subset_dataset(val_ds, subset_size=data_config['subset_size'])

    # get loaders
    loader_config = optional_fetch(data_config, 'loader')
    general_loader_config = optional_fetch(loader_config, 'general')
    train_loader = custom_loader(
        train_ds,
        verbose=verbose,
        **priority_merge_config(optional_fetch(loader_config, 'train'), general_loader_config)
    )
    val_loader = custom_loader(
        val_ds,
        verbose=verbose,
        **priority_merge_config(optional_fetch(loader_config, 'val'), general_loader_config)
    )
    test_loader = custom_loader(
        test_ds,
        verbose=verbose,
        **priority_merge_config(optional_fetch(loader_config, 'test'), general_loader_config)
    )

    return train_loader, val_loader, test_loader


def fetch_optimizer(
        model_params,
        optimizer_config: typing.Dict[str, typing.Any]
):
    """
    Function that creates optimizer based on the passed parameters in the dictionary.

    :param model_params: Model parameters to use for the creation of the optimizer
    :param optimizer_config: Config storing the config for the optimizer
    :type optimizer_config: typing.Dict[str, typing.Any]
    :return: Optimizer
    """
    optimizer = OPTIMIZER_DICT.get(optimizer_config['type'])
    return optimizer(model_params, **optimizer_config)


def fetch_scheduler(
        optimizer,
        scheduler_config: typing.Dict[str, typing.Any]
):
    """
    Function to build the scheduler. A scheduler adapts the learning rate to safely converge to an optimum.

    :param optimizer: Optimizer to use for the creation of the scheduler
    :param scheduler_config: Config to pass to the scheduler and the type of scheduler to use.
    :type scheduler_config: typing.Dict[str, typing.Any]
    :return: Scheduler
    """
    scheduler = SCHEDULER_DICT.get(scheduler_config['type'])
    return scheduler(optimizer, **scheduler_config)


def fetch_data_size(
        train_loader: torch.utils.data.DataLoader
) -> typing.Tuple[int, int]:
    """
    Function to fetch the data sizes from an existing loader. Returns the node and edge dimension.

    :param train_loader: Loader to use for fetching the first element
    :type train_loader: torch.utils.data.DataLoader
    :return: node and edge dimension, respectively
    :rtype: typing.Tuple[int, int]
    """
    # get first element
    for batch in train_loader:
        break

    # get node feature dim
    node_dim = batch.x.shape[1]

    # get edge attribute dim
    edge_attr_dim = batch.edge_attr.shape[1] if 'edge_attr' in batch else None

    return node_dim, edge_attr_dim


def create_model(
        data_sizes: typing.Tuple[int, int],
        model_config: typing.Dict[str, typing.Any]
) -> torch.nn.Module:
    """
    Function to create the model based on the passed config.

    :param data_sizes: sizes of the data, node and edge dimension respectively
    :type data_sizes: typing.Tuple[int, int]
    :param model_config: config to use for building the model
    :type model_config: typing.Dict[str, typing.Any]
    :return: Model built using the config
    :rtype: Inherits torch.nn.Module
    """
    if model_config['mode'] == 'custom':
        return CustomModel(
            data_sizes=data_sizes,
            model_config=model_config
        )
    else:
        raise NotImplementedError("Currently no other models besides custom model")


def training_procedure(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        task_type,
        device,
        n_max_epochs: int,  # hyperopt framework may or may not interrupt
        loss_function: str,  # cross entropy or l1
        clip_grad_norm: bool = False,
        out_dim: int = 1,
        use_test_set: bool = False,
        report: bool = False,
        verbose: bool = True,
        **kwargs
):
    # initialize loss function
    if loss_function == 'l1':
        loss_function = l1_loss
    elif loss_function == 'cross_entropy':
        if task_type == "classification_multilabel":
            loss_function = partial(
                graphgym_cross_entropy_loss,
                is_multilabel=True
            )
        else:
            loss_function = graphgym_cross_entropy_loss
    else:
        raise NotImplementedError("only l1 and cross entropy loss implemented")

    # initialize metric storage
    metric_storage = []

    # initialize loading of dataframe if required
    if task_type == 'regression_rank' and hasattr(train_loader.dataset, 'dataframe'):
        df_dict = {
            'train': train_loader.dataset.dataframe,
            'val': val_loader.dataset.dataframe,
            'test': test_loader.dataset.dataframe
        }

    else:
        df_dict = defaultdict(lambda: None)

    for epoch in range(n_max_epochs):
        if verbose:
            print(f"\nEpoch: {epoch}")
        metric_dict = {}

        metric_dict = train_epoch(
            metric_dict,
            clip_grad_norm,
            device,
            df_dict,
            loss_function,
            model,
            optimizer,
            out_dim,
            scheduler,
            task_type,
            train_loader,
            verbose=verbose
        )

        # val
        metric_dict = eval_epoch(metric_dict, device, df_dict, loss_function, model, out_dim, task_type, val_loader, 'val', verbose=verbose)

        # reporting to ray train
        if report:
            train.report(metric_dict)

        # test
        if use_test_set:
            metric_dict = eval_epoch(metric_dict, device, df_dict, loss_function, model, out_dim, task_type, test_loader, 'test', verbose=verbose)

        # append metric dict to list
        metric_storage.append(metric_dict)

    # todo:
    #  - integration to hyperparameter opt framework
    #  - checkpointing

    # return metric dict
    return pd.DataFrame(metric_storage)


def train_epoch(
        metric_dict: dict,
        clip_grad_norm: bool,
        device: str,
        df_dict: dict,
        loss_function,
        model: torch.nn.Module,
        optimizer,
        out_dim: int,
        scheduler,
        task_type: str,
        train_loader,
        verbose: bool = True
):
    model.train()

    loss_storage = []
    pred_storage = []
    true_storage = []

    # train, val
    if verbose:
        iter_loop = tqdm(train_loader)
    else:
        iter_loop = train_loader
    for batch in iter_loop:
        # reset optimizer
        optimizer.zero_grad()

        # put batch to device
        batch = batch.to(device)

        # generate prediction
        pred, true = model(batch)

        # calculate loss
        loss, pred = loss_function(pred, true)

        # train model
        loss.backward()

        # grad norm clipping
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # optimizer step
        optimizer.step()

        # add pred, true and loss to storage to compute metrics
        loss_storage.append(loss.item())
        pred_storage.append(pred.detach().cpu())
        true_storage.append(true.detach().cpu())
    # calc metrics (pred is modified for binary results)
    mode = 'train'  # todo: change when copying
    metric_dict.update(metric_system(
        pred=pred_storage,
        true=true_storage,
        task_type=task_type,
        device=device,
        training_mode=mode,
        dataframe=df_dict[mode],
        out_dim=out_dim,
        prefix=mode
    ))
    metric_dict[f'{mode}_mean_loss'] = sum(loss_storage) / len(loss_storage)
    metric_dict[f'{mode}_sum_loss'] = sum(loss_storage)
    # scheduler step
    scheduler.step()
    return metric_dict


def eval_epoch(
        metric_dict: dict,
        device: str,
        df_dict: dict,
        loss_function,
        model: torch.nn.Module,
        out_dim: int,
        task_type: str,
        eval_loader,
        mode: str,
        verbose: bool = True
):
    model.eval()

    loss_storage = []
    pred_storage = []
    true_storage = []

    # eval
    with torch.no_grad():
        iter_loop = tqdm(eval_loader) if verbose else eval_loader
        for batch in iter_loop:

            # put batch to device
            batch = batch.to(device)

            # generate prediction
            pred, true = model(batch)

            # calculate loss
            loss, pred = loss_function(pred, true)

            # add pred, true and loss to storage to compute metrics
            loss_storage.append(loss.item())
            pred_storage.append(pred.detach().cpu())
            true_storage.append(true.detach().cpu())

    # calc metrics (pred is modified for binary results)
    metric_dict.update(metric_system(
        pred=pred_storage,
        true=true_storage,
        task_type=task_type,
        device=device,
        training_mode=mode,
        dataframe=df_dict[mode],
        out_dim=out_dim,
        prefix=mode
    ))
    metric_dict[f'{mode}_mean_loss'] = sum(loss_storage) / len(loss_storage)
    metric_dict[f'{mode}_sum_loss'] = sum(loss_storage)
    return metric_dict


def run_config(
        config_dict: dict,
        report: bool = False,
        verbose: bool = True,
        device: str = None
):
    # seed everything
    seed = config_dict['seed'] if 'seed' in config_dict else 1
    torch_geometric.seed_everything(seed)

    # load data
    train_loader, val_loader, test_loader = fetch_loaders(config_dict['data'], verbose=verbose)

    # get model
    # todo: add intermediary layers (what to do with GPS? which version?)
    model = create_model(data_sizes=fetch_data_size(train_loader), model_config=config_dict['model'])

    # put model to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # get optimizer
    optimizer = fetch_optimizer(
        model.parameters(),
        config_dict['optimizer']
    )
    scheduler = fetch_scheduler(
        optimizer,
        config_dict['scheduler']
    )

    # start training
    metric_dict = training_procedure(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        out_dim=config_dict['model']['out_dim'] if 'out_dim' in config_dict['model'] else 1,
        report=report,
        verbose=verbose,
        **config_dict['training']
    )

    # todo: check functions
    return metric_dict
