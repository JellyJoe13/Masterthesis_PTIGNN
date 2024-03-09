import torch.cuda
from tqdm import tqdm

from ptgnn.dataset import DATASET_DICT
from ptgnn.loading.load import UniversalLoader
from ptgnn.loading.subsetting import subset_dataset
from ptgnn.model.framework.custom_model import CustomModel
from ptgnn.optimizing import OPTIMIZER_DICT, SCHEDULER_DICT
from ptgnn.runtime_config.config import priority_merge_config, optional_fetch
from ptgnn.runtime_config.loss import l1_loss, graphgym_cross_entropy_loss


def fetch_loaders(data_config: dict):
    dataset_config = data_config['dataset']
    # load dataset
    ds_type = DATASET_DICT.get(dataset_config['type'])
    train_ds = ds_type(**dataset_config, split='train')
    test_ds = ds_type(**dataset_config, split="test")
    val_ds = ds_type(**dataset_config, split="val")

    # subset data
    if 'subset_size' in dataset_config:
        train_ds = subset_dataset(train_ds, subset_size=data_config['subset_size'])
        test_ds = subset_dataset(test_ds, subset_size=data_config['subset_size'])
        val_ds = subset_dataset(val_ds, subset_size=data_config['subset_size'])

    # get loaders
    loader_config = optional_fetch(data_config, 'loader')
    general_loader_config = optional_fetch(loader_config, 'general')
    train_loader = UniversalLoader(
        train_ds,
        **priority_merge_config(optional_fetch(loader_config, 'train'), general_loader_config)
    )
    val_loader = UniversalLoader(
        val_ds,
        **priority_merge_config(optional_fetch(loader_config, 'val'), general_loader_config)
    )
    test_loader = UniversalLoader(
        test_ds,
        **priority_merge_config(optional_fetch(loader_config, 'test'), general_loader_config)
    )

    return train_loader, val_loader, test_loader


def fetch_optimizer(model_params, optimizer_config: dict):
    optimizer = OPTIMIZER_DICT.get(optimizer_config['type'])
    return optimizer(model_params, **optimizer_config)


def fetch_scheduler(optimizer, scheduler_config: dict):
    scheduler = SCHEDULER_DICT.get(scheduler_config['type'])
    return scheduler(optimizer, **scheduler_config)


def fetch_data_size(train_loader):
    # get first element
    for batch in train_loader:
        break

    # get node feature dim
    node_dim = batch.x.shape[1]

    # get edge attribute dim
    edge_attr_dim = batch.edge_attr.shape[1] if 'edge_attr' in batch else None

    return node_dim, edge_attr_dim


def create_model(data_sizes, model_config):
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
        metric_calculator,
        device,
        n_max_epochs: int,  # hyperopt framework may or may not interrupt
        loss_function: str,  # cross entropy or l1
        clip_grad_norm: bool = False,
        **kwargs
):
    # initialize loss function
    if loss_function == 'l1':
        loss_function = l1_loss
    elif loss_function == 'cross_entropy':
        loss_function = graphgym_cross_entropy_loss
    else:
        raise NotImplementedError("only l1 and cross entropy loss implemented")

    # initialize metric storage
    metric_storage_train = []
    metric_storage_val = []
    metric_storage_test = []

    for epoch in range(n_max_epochs):

        model.train()

        loss_storage = []
        pred_storage = []
        true_storage = []

        # train, val
        for batch in tqdm(train_loader):
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
        metric_dict = metric_calculator(
            pred.detach().cpu(),
            batch.y.detach().cpu()
        )
        metric_dict['loss'] = loss.item()

        # append metric dict to list
        metric_storage_train.append(metric_dict)

        # scheduler step
        scheduler.step()

        # val
        # todo copy paste with nograd und model.eval()
        # todo: report to ray and checkpointing
        # test
        # todo copy paste with nograd und model.eval()
        #  logic for if it should be done or not (tuning phase or final)

    # todo:
    #  check if final test (do evaluation on test_loader or not)
    #  generate metric scores
    #  integration to hyperparameter opt framework
    #  - reporting
    #  - checkpointing
    pass


def run_config(config_dict: dict):
    # todo: seed everything

    # load data
    train_loader, val_loader, test_loader = fetch_loaders(config_dict['data'])

    # get model
    # todo: add intermediary layers (what to do with GPS? which version?)
    model = create_model(data_sizes=fetch_data_size(train_loader), model_config=config_dict['model'])

    # put model to device
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

    # fetch metric calculators
    metric_calculator = ...  # todo

    # start training
    metric_dict = training_procedure(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metric_calculator=metric_calculator,
        device=device,
        **config_dict['training']
    )

    # save/display metrics
    # todo
    pass