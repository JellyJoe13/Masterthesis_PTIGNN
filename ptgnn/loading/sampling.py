import math
from itertools import chain
import random

import numpy as np
import torch.utils.data


class SingleConformerSampler(torch.utils.data.Sampler):
    """
    Adapted from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/submodules/ChIRo/model/datasets_samplers.py#L261
    to reproduce training procedure to some extent
    """

    def __init__(
            self,
            single_conformer_ds,
            full_ds,
            batch_size: int,
            n_pos: int = 0,
            n_neg: int = 1,
            without_replacement: bool = True,
            stratified: bool = True
    ):
        # set internal params
        self.single_conformer_ds = single_conformer_ds
        self.full_ds = full_ds
        self.batch_size = batch_size
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.without_replacement = without_replacement
        self.stratified = stratified

        # positive and negative sampler
        self.positive_sampler = SampleMapToPositives(self.full_ds, include_anchor=True)
        self.negative_sampler = SampleMapToNegatives(self.full_ds)

    def __iter__(self):
        # define groups
        groups = [
            [
                *self.positive_sampler.sample(
                    i,
                    n=1 + self.n_pos,
                    without_replacement=self.without_replacement
                ),
                *self.negative_sampler.sample(
                    i,
                    n=self.n_neg,
                    without_replacement=self.without_replacement,
                    stratified=self.stratified
                )
            ]
            for i in self.single_conformer_ds.index.values
        ]

        # shuffle single conformer wise groups
        np.random.shuffle(groups)

        # create batches
        batches = [
            list(chain(
                *groups[self.batch_size * i: self.batch_size * i + self.batch_size]
            ))
            for i in range(math.floor(len(groups) / self.batch_size))
        ]
        return iter(batches)


class SampleMapToPositives:
    """
    Adapted/copied from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/submodules/ChIRo/model/datasets_samplers.py#L167
    """
    def __init__(
            self,
            dataframe,
            is_sorted: bool = True,
            include_anchor: bool = False
    ):
        # isSorted vastly speeds up processing, but requires that the dataframe is sorted by SMILES_nostereo
        self.mapping = {}
        self.include_anchor = include_anchor

        for row_index, row in dataframe.iterrows():
            if is_sorted:
                subset_df = dataframe.iloc[max(row_index-50, 0): row_index+50, :]

                if self.include_anchor:
                    positives = set(subset_df[(subset_df.ID == row.ID)].index)
                else:
                    positives = set(subset_df[(subset_df.ID == row.ID) & (subset_df.index.values != row_index)].index)

                self.mapping[row_index] = positives

    def sample(
            self,
            i: int,
            n: int = 1,
            without_replacement: bool = True
    ):
        # sample positives
        if without_replacement:
            samples = random.sample(self.mapping[i], min(n, len(self.mapping[i])))
        else:
            samples = [random.choice(list(self.mapping[i])) for _ in range(n)]

        return samples


class SampleMapToNegatives:
    """
    Adapted/copied from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/submodules/ChIRo/model/datasets_samplers.py#L191
    """
    def __init__(
            self,
            dataframe,
            is_sorted: bool = True
    ):
        # isSorted vastly speeds up processing, but requires that the dataframe is sorted by SMILES_nostereo
        self.mapping = {}

        for row_index, row in dataframe.iterrows():

            if is_sorted:
                subset_df = dataframe.iloc[max(row_index-200, 0): row_index+200, :]

                grouped_negatives = subset_df[
                    (subset_df.SMILES_nostereo == row.SMILES_nostereo) & (subset_df.ID != row.ID)
                ].groupby(
                    by='ID',
                    sort=False
                ).groups.values()

                negative_classes = [
                    set(list(group))
                    for group in grouped_negatives
                ]

                self.mapping[row_index] = negative_classes

    def sample(
            self,
            i: int,
            n: int = 1,
            without_replacement: bool = True,
            stratified: bool = True
    ):
        # sample negatives
        if without_replacement:
            if stratified:
                samples = [
                    random.sample(self.mapping[i][j], min(len(self.mapping[i][j]), n))
                    for j in range(len(self.mapping[i]))
                ]
                samples = list(chain(*samples))
            else:
                population = list(chain(*[
                    list(self.mapping[i][j])
                    for j in range(len(self.mapping[i]))
                ]))
                samples = random.sample(population, min(len(population), n))

        else:
            if stratified:
                samples = [
                    [
                        random.choice(list(population))
                        for _ in range(n)
                    ]
                    for population in self.mapping[i]
                ]
                samples = list(chain(*samples))

            else:
                population = list(chain(*[
                    list(self.mapping[i][j])
                    for j in range(len(self.mapping[i]))
                ]))
                samples = [
                    random.choice(population)
                    for _ in range(n)
                ]

        return samples
