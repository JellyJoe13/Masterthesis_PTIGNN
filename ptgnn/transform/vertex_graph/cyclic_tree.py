import json

import torch
import torch_geometric.data


def cyclic_tree_vertex(
        data: torch_geometric.data.Data,
        mol_example
) -> torch_geometric.data.Data:
    """
    Adds cyclic vertices and edges to vertex graph

    :param data: vertex graph
    :type data: torch_geometric.data.Data
    :param mol_example: rdkit molecule of the data
    :return: modified data (in place though)
    :rtype: torch_geometric.data.Data
    """

    # getting rings
    atom_rings = list(mol_example.GetRingInfo().AtomRings())

    # define index offset
    offset = data.x.shape[0]

    # iterate over rings
    for idx, ring in enumerate(atom_rings):

        # prepare tree
        ptree_ring = {
            "C": [
                list(ring)
            ]
        }

        # add node
        data.x = torch.cat([
            data.x,
            torch.zeros(1, data.x.shape[1])
        ], dim=0)

        ptree_neighbor_rings = []

        # check with connection to other trees
        for o_idx, other_rings in enumerate(atom_rings):
            if idx == o_idx:
                continue

            # check if connected
            intersection = list(set(ring) & set(other_rings))

            if len(intersection):
                ptree_neighbor_rings.append(offset + o_idx)

        if len(ptree_neighbor_rings):
            data.ptree.append(json.dumps({
                "S": [
                    ptree_ring,
                    {
                        "P": ptree_neighbor_rings
                    }
                ]
            }))
        else:
            data.ptree.append(json.dumps(ptree_ring))

    # iterate over nodes and find out whether they belong to a ring (and should thus be connected to the ring node)
    for i in range(offset):

        temp_tree = json.loads(data.ptree[i])

        if "S" in temp_tree:
            # get rings it is in
            in_rings = [
                offset + idx
                for idx, r in enumerate(atom_rings)
                if i in r
            ]

            if len(in_rings):
                temp_tree["S"].append({
                    "P": in_rings
                })
        data.ptree[i] = json.dumps(temp_tree)

    return data
