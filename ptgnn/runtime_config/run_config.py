from ptgnn.dataset import DATASET_DICT
from ptgnn.loading.load import UniversalLoader
from ptgnn.loading.subsetting import subset_dataset
from ptgnn.model.framework.custom_model import CustomModel
from ptgnn.optimizing import OPTIMIZER_DICT, SCHEDULER_DICT
from ptgnn.runtime_config.config import priority_merge_config, optional_fetch


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


def run_config(config_dict: dict):
    # load data
    train_loader, val_loader, test_loader = fetch_loaders(config_dict['data'])

    # get model
    # todo
    model = create_model(data_sizes=fetch_data_size(train_loader), model_config=config_dict['model'])

    # get optimizer
    optimizer = fetch_optimizer(
        model.parameters(),
        config_dict['optimizer']
    )
    scheduler = fetch_scheduler(
        optimizer,
        config_dict['scheduler']
    )

    # fetch training mode
    # todo
    # fetch metrics (combined with above) (or inserted)
    # todo
    pass