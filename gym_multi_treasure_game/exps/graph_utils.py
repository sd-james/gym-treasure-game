import networkx as nx
import numpy as np

from s2s.utils import load

KNN = load('/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/data/knn.pkl')


def extract_options(edge):
    if edge['level'] == 0:
        return [edge['action'].option]

    raise NotImplementedError


def shortest_path_edges(graph, source, target):
    """
    Return the edges along the shortest path from source to target, or None if no path exists
    """
    if not nx.has_path(graph, source, target):
        return None
    path = nx.shortest_path(graph, source, target)
    path_graph = nx.path_graph(path)
    # Read attributes from each edge
    plan = list()
    for ea in path_graph.edges():
        edge = graph.edges[ea[0], ea[1]]
        plan.append(edge)
    return plan


def _combine(state, inv):
    """
    Convert the inventory image into  a simple vector and combine with the state info
    """
    global KNN

    has_key = 0
    has_gold = 0

    temp = tuple(np.around(inv.astype(float), 3))
    ret = KNN.predict([temp])[0]
    if ret == 1:
        has_key = 1
    elif ret == 2:
        has_gold = 1
    return np.hstack((state, [has_key, has_gold]))


def find_nodes(state, graph):
    """
    Return the nodes in the graph that are likely representative of the given state
    """

    def _dist(a, b):
        m = min(len(a), len(b))
        a = a[0:m]
        b = b[0:m]
        for i in range(len(a)):
            if abs(a[i] - b[i]) > 0.1:
                return np.inf
        return np.linalg.norm(a - b)

    if len(state.shape) == 2:
        state = state.squeeze()
    candidates = list()
    best_dist = np.inf
    for node in graph.nodes:
        s = graph.nodes[node]['state']
        if len(s.shape) == 2:
            s = s.squeeze()
        s = _combine(s, graph.nodes[node]['obs'][1])
        dist = _dist(s, state)

        if dist < best_dist:
            best_dist = dist
            candidates.append(node)
        elif not np.isinf(dist) and abs(best_dist - dist) < 0.000001:
            candidates.append(node)
    return candidates
