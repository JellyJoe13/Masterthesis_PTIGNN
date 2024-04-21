import pandas as pd

from ptgnn.loading.load import UniversalLoader
from ptgnn.loading.sampling import SingleConformerSampler


def custom_loader(
        dataset,
        n_neighbors_in_circle: int,
        batch_size: int,
        sampler: str = "full_batch",  # alternative: single_conformer_sampler
        shuffle: bool = True,
        dataframe: pd.DataFrame = None,
        num_workers: int = 0,
        verbose: bool = True,
        precompute_pos_enc: list = []
) -> UniversalLoader:
    """
    Fetches the loader for elements in project. Currently one single loader for all elements.

    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L99

    :param dataset: Dataset which to wrap in a loader
    :param n_neighbors_in_circle: number of neighbors in circle in case of ChiENN cycle index setting.
    :type n_neighbors_in_circle: int
    :param batch_size: batch size
    :type batch_size: int
    :param sampler: Whether to use the full batch or single conformer sampling method
    :type sampler: str
    :param shuffle: Whether to shuffle the data or not
    :type shuffle: bool
    :param dataframe: dataframe which to use in single conformer sampling
    :type dataframe: pd.DataFrame
    :param num_workers: How many workers to use in the loader
    :type num_workers: int
    :param verbose: Whether to print information to command line/system out
    :type verbose: bool
    :param precompute_pos_enc: List of positional embeddings to calculate for the contents in advance
    :type precompute_pos_enc: list
    """
    if sampler == "single_conformer_sampler":
        # select dataframe
        if dataframe is None:
            if not hasattr(dataset, 'dataframe'):
                raise Exception("set sampling to single conformer sampler but the dataset does not have a corresponding"
                                "dataframe.")
            dataframe = dataset.dataframe

        # suspicion that this parameter is utterly pointless if single_conformer is enabled in the datasets
        # because of this action:
        # group by id and randomly select one
        single_conformer_df = dataframe.groupby('ID').sample(1)

        # create sampler
        sampler = SingleConformerSampler(
            single_conformer_ds=single_conformer_df,
            full_ds=dataframe,
            batch_size=batch_size,
            n_pos=0,
            n_neg=1,
            without_replacement=True,
            stratified=True
        )

        # create loader and return
        return UniversalLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            n_neighbors_in_circle=n_neighbors_in_circle,
            verbose=verbose,
            precompute_pos_enc=precompute_pos_enc
        )

    # not sure for what the original authors included the latter trigger condition
    elif sampler == "full_batch" or len(dataset) > 1:
        return UniversalLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            n_neighbors_in_circle=n_neighbors_in_circle,
            verbose=verbose,
            precompute_pos_enc=precompute_pos_enc
        )

    else:
        raise NotImplementedError(f"No sampler mode {sampler}. Choose full_batch or single_conformer_sampler instead.")
