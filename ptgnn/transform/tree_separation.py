import json

import torch
import torch_geometric.data


def separate_tree_into_subtrees(
        data_elem: torch_geometric.data.Data
) -> torch_geometric.data.Data:
    """
    Separates the permutation trees by introducing new nodes and substituting subtrees in trees with new nodes.
    Initialized nodes with indices. Edge index is not changed as not used in Permutation tree model. Note that operation
    is IN-PLACE.

    :param data_elem: data element which is to be split/transformed
    :type data_elem: torch_geometric.data.Data
    :return: modified element. Operation is in place, element is returned in case in-place property is removed later on
    :rtype: torch_geometric.data.Data
    """
    current_idx = 0

    while current_idx < data_elem.x.shape[0]:
        # get tree
        ptree = data_elem.ptree[current_idx]

        # convert tree into dict
        ptree = json.loads(ptree)

        # iterate over subtrees
        key = next(iter(ptree.keys()))
        for idx, subtree in enumerate(ptree[key]):
            if isinstance(subtree, int):
                # this means that element is already a leaf node
                continue

            elif isinstance(subtree, dict):
                # element is a tree
                # replace tree with new_idx
                ptree[key][idx] = data_elem.x.shape[0]

                # append zeros to x
                data_elem.x = torch.cat([data_elem.x, torch.zeros(1, data_elem.x.shape[1])], dim=0)

                # append new ptree
                data_elem.ptree.append(json.dumps(subtree))

            else:
                raise Exception(f"Something went wrong, {subtree} is neither int nor dict.")

        # save ptree
        data_elem.ptree[current_idx] = json.dumps(ptree)

        # increase current idx
        current_idx += 1

    return data_elem
