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
        num_workers: int = 0
) -> UniversalLoader:
    """
    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/utils.py#L99
    """
    if sampler == "single_conformer_sampler":
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
            n_neighbors_in_circle=n_neighbors_in_circle
        )

    # not sure for what the original authors included the latter trigger condition
    elif sampler == "full_batch" or len(dataset) > 1:
        return UniversalLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            n_neighbors_in_circle=n_neighbors_in_circle
        )

    else:
        raise NotImplementedError(f"No sampler mode {sampler}. Choose full_batch or single_conformer_sampler instead.")
