import json
import typing

import torch
import torch_geometric


def cyclic_tree_edge(
        data: torch_geometric.data.Data,
        mol_example,
        node_mapping: typing.Dict[str, typing.Any]
) -> torch_geometric.data.Data:
    """
    Adds cyclic vertices and edges to edge graph

    :param data: edge graph
    :type data: torch_geometric.data.Data
    :param mol_example: rdkit molecule of the data
    :param node_mapping: Map from (node, node) to edge index
    :type node_mapping: typing.Dict[str, typing.Any]
    :return: modified data (in place though)
    :rtype: torch_geometric.data.Data
    """
    # getting rings
    atom_rings = list(mol_example.GetRingInfo().AtomRings())

    # compute offset
    offset = data.x.shape[0]

    for idx, ring in enumerate(atom_rings):
        # convert ring into edge ring
        edge_ring = list(zip(
            ring,
            ring[1:] + ring[:1]
        ))
        edge_idx_ring_a = [
            node_mapping[(a,b)]
            for a,b in edge_ring
            if (a,b) in node_mapping
        ]
        edge_idx_ring_b = [
            node_mapping[(b,a)]
            for a,b in edge_ring
            if (b,a) in node_mapping
        ]

        # prepare tree
        if len(edge_idx_ring_a) and len(edge_idx_ring_b):
            ptree_ring = {
                "P": [{
                    "C": list(edge_idx_ring_a)
                }, {
                    "C": list(edge_idx_ring_b)
                }]
            }
        elif len(edge_idx_ring_a):
            ptree_ring = {
                "C": edge_idx_ring_a
            }
        elif len(edge_idx_ring_b):
            ptree_ring = {
                "C": edge_idx_ring_b
            }
        else:
            continue

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

    # create reverse mapping
    reverse_dict = dict(zip(node_mapping.values(), node_mapping.keys()))

    for i in range(offset):
        source, target = reverse_dict[i]

        temp_tree = json.loads(data.ptree[i])
        if "S" in temp_tree:
            # get rings it is in
            in_rings = [
                offset + idx
                for idx, r in enumerate(atom_rings)
                if source in r and target in r
            ]

            if len(in_rings):
                temp_tree["S"].append({
                    "P": in_rings
                })
        data.ptree[i] = json.dumps(temp_tree)

    return data


def cyclic_tree_edge_light(
        data: torch_geometric.data.Data,
        mol_example,
        node_mapping: typing.Dict[str, typing.Any]
) -> torch_geometric.data.Data:
    """
    Adds cyclic vertices and edges to edge graph. Light version: no neighboring cycles

    :param data: edge graph
    :type data: torch_geometric.data.Data
    :param mol_example: rdkit molecule of the data
    :param node_mapping: Map from (node, node) to edge index
    :type node_mapping: typing.Dict[str, typing.Any]
    :return: modified data (in place though)
    :rtype: torch_geometric.data.Data
    """
    # getting rings
    atom_rings = list(mol_example.GetRingInfo().AtomRings())

    # compute offset
    offset = data.x.shape[0]

    for idx, ring in enumerate(atom_rings):
        # convert ring into edge ring
        edge_ring = list(zip(
            ring,
            ring[1:] + ring[:1]
        ))
        edge_idx_ring_a = [
            node_mapping[(a,b)]
            for a,b in edge_ring
            if (a,b) in node_mapping
        ]
        edge_idx_ring_b = [
            node_mapping[(b,a)]
            for a,b in edge_ring
            if (b,a) in node_mapping
        ]

        # prepare tree
        if len(edge_idx_ring_a) and len(edge_idx_ring_b):
            ptree_ring = {
                "P": [{
                    "C": list(edge_idx_ring_a)
                }, {
                    "C": list(edge_idx_ring_b)
                }]
            }
        elif len(edge_idx_ring_a):
            ptree_ring = {
                "C": edge_idx_ring_a
            }
        elif len(edge_idx_ring_b):
            ptree_ring = {
                "C": edge_idx_ring_b
            }
        else:
            continue

        # add node
        data.x = torch.cat([
            data.x,
            torch.zeros(1, data.x.shape[1])
        ], dim=0)

        data.ptree.append(json.dumps(ptree_ring))

    # create reverse mapping
    reverse_dict = dict(zip(node_mapping.values(), node_mapping.keys()))

    for i in range(offset):
        source, target = reverse_dict[i]

        temp_tree = json.loads(data.ptree[i])
        if "S" in temp_tree:
            # get rings it is in
            in_rings = [
                offset + idx
                for idx, r in enumerate(atom_rings)
                if source in r and target in r
            ]

            if len(in_rings):
                temp_tree["S"].append({
                    "P": in_rings
                })
        data.ptree[i] = json.dumps(temp_tree)

    return data


def cyclic_tree_edge_minimal(
        data: torch_geometric.data.Data,
        mol_example,
        node_mapping: typing.Dict[str, typing.Any]
) -> torch_geometric.data.Data:
    """
    Adds cyclic vertices and edges to edge graph. Minimal version: no neighboring cycles and no connection to cylce
    in participating nodes/edges

    :param data: edge graph
    :type data: torch_geometric.data.Data
    :param mol_example: rdkit molecule of the data
    :param node_mapping: Map from (node, node) to edge index
    :type node_mapping: typing.Dict[str, typing.Any]
    :return: modified data (in place though)
    :rtype: torch_geometric.data.Data
    """
    # getting rings
    atom_rings = list(mol_example.GetRingInfo().AtomRings())

    # compute offset
    offset = data.x.shape[0]

    for idx, ring in enumerate(atom_rings):
        # convert ring into edge ring
        edge_ring = list(zip(
            ring,
            ring[1:] + ring[:1]
        ))
        edge_idx_ring_a = [
            node_mapping[(a,b)]
            for a,b in edge_ring
            if (a,b) in node_mapping
        ]
        edge_idx_ring_b = [
            node_mapping[(b,a)]
            for a,b in edge_ring
            if (b,a) in node_mapping
        ]

        # prepare tree
        if len(edge_idx_ring_a) and len(edge_idx_ring_b):
            ptree_ring = {
                "P": [{
                    "C": list(edge_idx_ring_a)
                }, {
                    "C": list(edge_idx_ring_b)
                }]
            }
        elif len(edge_idx_ring_a):
            ptree_ring = {
                "C": edge_idx_ring_a
            }
        elif len(edge_idx_ring_b):
            ptree_ring = {
                "C": edge_idx_ring_b
            }
        else:
            continue

        # add node
        data.x = torch.cat([
            data.x,
            torch.zeros(1, data.x.shape[1])
        ], dim=0)

        data.ptree.append(json.dumps(ptree_ring))

    return data
