import json

import torch


def get_cistrans_tree(vertex_graph, molecule, node_a, node_b, node_mapping):
    # get neighbors
    neighbors = vertex_graph.edge_index[1, vertex_graph.edge_index[0] == node_a]
    neighbors = neighbors[neighbors != 4]

    neighbors = torch.cat([neighbors, vertex_graph.edge_index[1, vertex_graph.edge_index[0] == node_b]])
    neighbors = neighbors[neighbors != 3].sort().values

    # get positions of neighbors
    pos = vertex_graph.pos[neighbors]

    # calculate distances with 2-norm
    distances = torch.zeros(4,4)
    for i in range(4):
        for j in range(4):
            distances[i, j] = ((pos[i] - pos[j]) ** 2).sum().sqrt()

    def calc_order(distances):
        order = [0]
        while len(distances):
            rel_order = distances[order[-1]].argsort()
            for i in range(4):
                if not rel_order[i] in order:
                    order.append(int(rel_order[i]))
                    break
            else:
                return torch.tensor(order)

    # calculate order by getting the neirest of the neighbors, then the next until finished
    order = calc_order(distances)
    neighbor_order = neighbors[order]

    # current and reference node
    node_list = [node_mapping[(node_a, node_b)]]
    node_list += [node_mapping[(node_b, node_a)]]
    # create tree and return it
    return json.dumps({
        "S": [
            node_list + [
                {"C": neighbor_order.tolist()}
            ]
        ]
    })
