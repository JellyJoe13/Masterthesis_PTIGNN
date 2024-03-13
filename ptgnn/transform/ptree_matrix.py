import json
import torch
import typing


type_dict = {
    "P": 1,
    "Z": 2,
}


def permutation_tree_depth(tree):
    if isinstance(tree, list):
        return max(map(permutation_tree_depth, tree))
    if isinstance(tree, dict):
        return 1 + (max(map(permutation_tree_depth, tree.values())) if tree else 0)
    return 0


def get_matrix(tree, depth, idx_prefix: list = [], type_prefix: list = []) -> list:
    if isinstance(tree, list):
        idx_matrix, type_matrix = zip(*[
            get_matrix(child, depth, idx_prefix + [idx], type_prefix)
            for idx, child in enumerate(tree)
        ])
        return torch.cat(idx_matrix, dim=0), torch.cat(type_matrix, dim=0)

    elif isinstance(tree, dict):
        key = next(iter(tree.keys()))
        return get_matrix(tree[key], depth - 1, idx_prefix, type_prefix=type_prefix + [type_dict[key]])

    elif isinstance(tree, int):

        return torch.tensor(idx_prefix + [tree] + [-1] * depth).reshape(1, -1), \
            torch.tensor(type_prefix + [0]*depth).reshape(1, -1)

    else:
        return [-1]


def permutation_tree_to_matrix(ptree_string_list: typing.List[str], k: int = 3):
    # transform to dict
    permutation_trees = [
        json.loads(p_string)
        for p_string in ptree_string_list
    ]

    # get number of layers
    num_layers = permutation_tree_depth(permutation_trees)

    # fetch the matrix for the permutation trees
    return get_matrix(permutation_trees, depth=num_layers)
