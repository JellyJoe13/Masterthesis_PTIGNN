from typing import Union, List, Optional

from torch.utils.data import DataLoader
from torch_geometric.config_store import Dataset
from torch_geometric.data.data import BaseData

from ptgnn.loading.collate import UniversalCollater
from ptgnn.loading.pos_enc import precompute_pos_enc_function


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
            precompute_pos_enc: list = [],
            verbose: bool = True,
            **kwargs
    ):
        """
        Universal Loader init function. Capable of loading all data types/datasets in this project.

        :param dataset: dataset which to load with the loader
        :type dataset: Union[Dataset, List[BaseData]]
        :param batch_size: batch size
        :type batch_size: int
        :param shuffle: Whether or not to shuffle the dataset
        :type shuffle: bool
        :param follow_batch: Follow batch parameter of Collater
        :type follow_batch: Optional[List[str]]
        :param exclude_keys: Exclude keys parameter of Collater
        :type exclude_keys: Optional[List[str]]
        :param n_neighbors_in_circle: number of neighbors in circle - relevant for ChiENN model
        :type n_neighbors_in_circle: Optional[int]
        :param precompute_pos_enc: list of positional embeddings to compute
        :type precompute_pos_enc: list
        :param verbose: Whether or not to print intermediary output/progress
        :type verbose: bool
        :param kwargs: Excessive parameters
        """
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

        if len(precompute_pos_enc):
            precompute_pos_enc_function(dataset, precompute_pos_enc, verbose=verbose)
