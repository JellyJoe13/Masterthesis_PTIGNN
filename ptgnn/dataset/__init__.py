from .mc_dataset import MCDataset
from .rs_dataset import RSDataset
from .bindingaffinity_dataset import BindingAffinityDataset
from .csv_datasets import BaceDataset, Tox21Dataset
from .ogb_datasets import OGBDataset
from .ez_dataset import EZDataset

DATASET_DICT = {
    "rs": RSDataset,
    "ez": EZDataset,
    "ba": BindingAffinityDataset,
    "bace": BaceDataset,
    "tox21": Tox21Dataset,
    "ogb": OGBDataset,
    "mc": MCDataset
}
