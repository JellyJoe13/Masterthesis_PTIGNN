"""
This file is written for fixing issues with scaffoldsplit in the library chainer-chemistry. Part of the code from there
has been copied. Sources:
https://chainer-chemistry.readthedocs.io/en/latest/generated/chainer_chemistry.dataset.splitters.ScaffoldSplitter.html
https://chainer-chemistry.readthedocs.io/en/latest/_modules/chainer_chemistry/dataset/splitters/scaffold_splitter.html#ScaffoldSplitter
"""
from collections import defaultdict

import numpy as np
from chainer_chemistry.dataset.splitters.scaffold_splitter import generate_scaffold


def scaffold_split(
        dataset,
        smiles_list,
        frac_train,
        frac_valid,
        frac_test,
        seed,
        include_chirality
):
    # check whether fractions tum to one
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

    # length check
    if len(dataset) != len(smiles_list):
        raise ValueError("The lengths of dataset and smiles_list are different")

    # use random generator with seed
    rng = np.random.RandomState(seed)

    # get scaffold split
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality)
        scaffolds[scaffold].append(ind)

    # generate permutation of scaffold groups
    # here's where the error is - putting list inside this does not work anymore
    scaffold_sets = list(scaffolds.values())
    permutation_idx = rng.permutation(np.arange(len(scaffold_sets)))
    scaffold_sets = [
        scaffold_sets[idx]
        for idx in permutation_idx
    ]

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_index = []
    valid_index = []
    test_index = []

    for scaffold_set in scaffold_sets:
        if len(valid_index) + len(scaffold_set) <= n_total_valid:
            valid_index.extend(scaffold_set)
        elif len(test_index) + len(scaffold_set) <= n_total_test:
            test_index.extend(scaffold_set)
        else:
            train_index.extend(scaffold_set)

    return np.array(train_index), np.array(valid_index), np.array(test_index)

