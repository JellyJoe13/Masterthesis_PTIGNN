import json
import torch
import typing


type_dict = {
    "P": 1,
    "Z": 2,
    "S": 3,
    "Z2": 4,
    "S2": 5
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


def remove_inverted_through_p(perm_tree: dict, use_new_inv: bool = False):
    if isinstance(perm_tree, int):
        return perm_tree
    elif isinstance(perm_tree, list):
        # no inverting happening here, this is meant for normal types
        return [remove_inverted_through_p(subtree, use_new_inv) for subtree in perm_tree]

    key = next(iter(perm_tree.keys()))
    if key == "Q":
        return {
            "P": [
                {
                    f"S{2 if use_new_inv else ''}": [
                        remove_inverted_through_p(subtree, use_new_inv)
                        for subtree in perm_tree[key]
                    ]
                },
                {
                    f"S{2 if use_new_inv else ''}": [
                             remove_inverted_through_p(subtree, use_new_inv)
                             for subtree in perm_tree[key]
                         ][::-1]
                }
            ]
        }
    elif key == "C":
        return {
            "P": [
                {
                    f"Z{2 if use_new_inv else ''}": [
                        remove_inverted_through_p(subtree, use_new_inv)
                        for subtree in perm_tree[key]
                    ]
                },
                {
                    f"Z{2 if use_new_inv else ''}": [
                             remove_inverted_through_p(subtree, use_new_inv)
                             for subtree in perm_tree[key]
                         ][::-1]
                }
            ]
        }
    else:
        return {key: remove_inverted_through_p(value, use_new_inv) for key, value in perm_tree.items()}


def permutation_tree_to_matrix(
        ptree_string_list: typing.List[str],
        k: int = 3,
        use_new_inv: bool = False
):
    # transform to dict
    permutation_trees = [
        json.loads(p_string)
        for p_string in ptree_string_list
    ]

    # replace C, Q by P(Z,Z) and P(S,S)
    permutation_trees = [
        remove_inverted_through_p(tree, use_new_inv)
        for tree in permutation_trees
    ]

    # get number of layers
    num_layers = permutation_tree_depth(permutation_trees)

    # fetch the matrix for the permutation trees
    idx_matrix, type_matrix = get_matrix(permutation_trees, depth=num_layers)

    # fill -1 in matrix with last values
    for i in range(1, idx_matrix.shape[1]):
        # get the -1 entry mask
        mask = idx_matrix[:, i] == -1
        idx_matrix[mask, i] = idx_matrix[mask, i-1]

    return idx_matrix, type_matrix


def permutation_tree_to_order_matrix(
        batch,
        k,
        use_new_inv: bool = False
):  # can also be data object as long as it has ptree argument
    # get matrices
    idx_matrix, type_matrix = permutation_tree_to_matrix(batch.ptree, k, use_new_inv)

    # get structure to orient to
    idx_structure = idx_matrix[:, :-1]

    for layer_idx in range(idx_matrix.shape[1]-2):
        # get indexes for graph pooling
        idx_structure, current_layer_pooling_counts = torch.unique(idx_structure[:, :-1], dim=0, return_counts=True)

        # get indexes for graph pooling
        current_layer_pooling = torch.repeat_interleave(current_layer_pooling_counts)

        # init circling
        order_matrix = torch.zeros(k, len(current_layer_pooling), dtype=torch.int) - 1

        # create new type_matrix
        new_type_matrix = []

        cur_pos = 0
        for i in current_layer_pooling_counts:
            current_k = min(k, i)

            # create the index to put into the order matrix
            r = torch.arange(cur_pos, cur_pos+i)

            # get type
            new_type_matrix.append(type_matrix[cur_pos, :-1])
            t = type_matrix[cur_pos, -1].item()

            # switch for node types
            if t == 0 or t == 1:
                order_matrix[0, cur_pos:(cur_pos+current_k)] = r[:current_k]
            elif t == 2:
                for j in range(i):
                    order_matrix[:current_k, cur_pos+j] = torch.roll(r, shifts=-j)[:current_k]
            elif t == 3:
                for j in range(i - current_k + 1):
                    order_matrix[:current_k, cur_pos+j] = torch.roll(r, shifts=-j)[:current_k]

            cur_pos += i

        # double list is a workaround for pyg auto collate function in datasets. Other way: hard coding tree depth
        setattr(batch, "initial_map", idx_matrix[:, -1])
        setattr(batch, f"layer{layer_idx}_order_matrix", [[order_matrix]])
        setattr(batch, f"layer{layer_idx}_type_mask", [[type_matrix[:, -1]]])
        setattr(batch, f"layer{layer_idx}_pooling", [[current_layer_pooling]])
        setattr(batch, f"num_layer", int(idx_matrix.shape[1]-2))

        type_matrix = torch.stack(new_type_matrix, dim=0)

    return batch

