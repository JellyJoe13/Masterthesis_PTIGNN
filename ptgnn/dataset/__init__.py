from .rs_dataset import RSDataset
from .bindingaffinity_dataset import BindingAffinityDataset
from .csv_datasets import BaceDataset, Tox21Dataset
from .ogb_datasets import OGBDataset

DATASET_DICT = {
    "rs": RSDataset,
    "ba": BindingAffinityDataset,
    "bace": BaceDataset,
    "tox21": Tox21Dataset,
    "ogb": OGBDataset
}
