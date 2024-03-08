from ptgnn.dataset import DATASET_DICT
from ptgnn.loading.load import UniversalLoader
from ptgnn.loading.subsetting import subset_dataset
from ptgnn.optimizing import OPTIMIZER_DICT, SCHEDULER_DICT


def fetch_loaders(data_config: dict):
    dataset_config = data_config['dataset']
    # load dataset
    ds_type = DATASET_DICT.get(dataset_config['ds_type'])
    train_ds = ds_type(**dataset_config, split='train')
    test_ds = ds_type(**dataset_config, split="test")
    val_ds = ds_type(**dataset_config, split="val")

    # subset data
    if 'subset_size' in dataset_config:
        train_ds = subset_dataset(train_ds, subset_size=data_config['subset_size'])
        test_ds = subset_dataset(test_ds, subset_size=data_config['subset_size'])
        val_ds = subset_dataset(val_ds, subset_size=data_config['subset_size'])

    # get loaders
    train_loader = UniversalLoader(train_ds, **data_config['loader']['train'], **data_config['loader']['general'])
    val_loader = UniversalLoader(val_ds, **data_config['loader']['val'], **data_config['loader']['general'])
    test_loader = UniversalLoader(test_ds, **data_config['loader']['test'], **data_config['loader']['general'])

    return train_loader, val_loader, test_loader


def fetch_optimizer(model_params, optimizer_config: dict):
    optimizer = OPTIMIZER_DICT.get(optimizer_config['type'])
    return optimizer(model_params, **optimizer_config)


def fetch_scheduler(optimizer, scheduler_config: dict):
    scheduler = SCHEDULER_DICT.get(scheduler_config['type'])
    return scheduler(optimizer, **scheduler_config)


def run_config(config_dict: dict):
    # load data
    train_loader, val_loader, test_loader = fetch_loaders(config_dict['data'])

    # get model
    # todo
    model = ...

    # get optimizer
    optimizer = fetch_optimizer(model.params(), **config_dict['optimizer'])
    scheduler = fetch_scheduler(optimizer, **config_dict['scheduler'])

    # fetch training mode
    # todo
    # fetch metrics (combined with above) (or inserted)
    # todo
    pass