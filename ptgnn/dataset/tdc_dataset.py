import typing
from torch_geometric.data import InMemoryDataset


class TDCDataset(InMemoryDataset):
    """
    Inspired/Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/tdc_dataset.py
    """
    def __init__(
            self,
            root: str = "src/tdc",
            tdc_type,
            tdc_ds_name,
            tdc_assay_name,
            mask_chiral_tags: bool = False,
            split: str = "train",
            graph_mode: str = "edge",
            transformation_mode: str = "default",
            transformation_parameters: typing.Dict[str, typing.Any] = {},
            max_atoms: int = 100,
            max_attempts: int = 100,  # significantly decreased - 5000 is way too much!
            min_number_of_chiral_centers: int = 0
    ):
        pass

    # apparently loads tox21 dataset but we already have that in csv datasets