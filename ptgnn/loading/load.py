from typing import Union, List, Optional

from torch.utils.data import DataLoader
from torch_geometric.config_store import Dataset
from torch_geometric.data.data import BaseData

from ptgnn.loading.collate import UniversalCollater


class UniversalLoader(DataLoader):
    """
    Inspired/adapted from https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/dataset/dataloader.py
    """
    def __init__(
            self,
            dataset: Union[Dataset, List[BaseData]],
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            n_neighbors_in_circle: Optional[int] = None,
            **kwargs
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=UniversalCollater(
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
                n_neighbors_in_circle=n_neighbors_in_circle
            ),
            **kwargs
        )
