import random

import pandas as pd


def dataset_select_indices(dataset, indices):
    # get list of data objects
    data_list = [dataset[i] for i in indices]

    # set data and slice to dataset
    dataset.data, dataset.slices = dataset.collate(data_list)

    return dataset


def subset_dataset(
        dataset,
        subset_size: int = 10_000,
):
    """
    Rework of code from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/loader/master_loader.py#L697

    Subselects a part of dataset to make hyperparameter optimization for huge datasets feasible.

    :param dataset: dataset to subselect
    :param subset_size: (maximal) size of the dataset after subsetting
    :type subset_size: int
    """
    if hasattr(dataset, 'dataframe'):
        # extract dataframe
        dataframe = dataset.dataframe

        # sample dataset with dataframe
        indices = sample_with_dataframe(dataframe, subset_size)

        # select dataframe
        dataframe = dataframe.iloc[indices, :].reset_index()

        # select dataset subset
        dataset = dataset_select_indices(dataset, indices)

        # update dataframe in dataset
        dataset.dataframe = dataframe

        return dataset

    else:
        # sample dataset without dataframe
        indices = sample_indices_simple(dataset, subset_size)

        # select dataset subset
        dataset = dataset_select_indices(dataset, indices)

        return dataset


def sample_indices_simple(
        dataset,
        subset_size: int
):
    """
    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/loader/master_loader.py#L862
    """
    if len(dataset) > subset_size:
        # do something
        return sorted(random.sample(range(len(dataset)), k=subset_size))
    else:
        # do nothing
        return list(range(len(dataset)))


def sample_with_dataframe(
        dataframe,
        subset_size: int
):
    """
    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/loader/master_loader.py#L872
    """
    if len(dataframe) > subset_size:
        # some curious check whether it exactly has 2 enantiomers for the dataframe...
        if len(dataframe) != len(dataframe["SMILES_nostereo"].unique()) * 2:
            raise ValueError(
                "Every molecule must have exactly one enantiomer in the dataframe!"
            )

        samples = sorted(random.sample(range(len(dataframe) // 2), k=subset_size // 2))
        indices = []
        for s in samples:
            indices.append(2 * s)
            indices.append(2 * s + 1)
        return indices
