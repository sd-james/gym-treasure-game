from collections import OrderedDict

import networkx as nx
import numpy as np

from s2s.hierarchy.option_discovery import _find_elbow
from s2s.utils import show


def _identify_subgoals(graph: nx.DiGraph, max_length=4, subgoal_method='betweenness', verbose=False,
                       **kwargs):
    # identify important states
    if subgoal_method == 'betweenness':
        ordered_subgoals = nx.algorithms.centrality.betweenness_centrality(graph)
        ordered_subgoals = OrderedDict(sorted(ordered_subgoals.items(), key=lambda kv: kv[1], reverse=True))

        if 'reduce_graph' in kwargs:
            graph_reducer = kwargs.get('reduce_graph', 3)
            num_nodes = len(graph.nodes) // graph_reducer
        else:
            num_nodes = _find_elbow(list(ordered_subgoals.values()))
        num_nodes = max(2, num_nodes)
        subgoals = dict()
        for count, (end_state_id, score) in enumerate(ordered_subgoals.items()):
            if count >= num_nodes:
                break
            subgoals[end_state_id] = score
    elif subgoal_method == 'voterank':

        graph_reducer = kwargs.get('reduce_graph', 3)
        num_nodes = len(graph.nodes) // graph_reducer
        num_nodes = max(2, num_nodes)
        targets = nx.algorithms.centrality.voterank(graph, num_nodes)
        subgoals = {x: 1 for x in targets}
    else:
        raise ValueError("No such method {}".format(subgoal_method))

    targets = list()
    paths = list()

    show("Identifed {} subgoals using {}".format(num_nodes, subgoal_method), verbose)

    for count, (end_state_id, score) in enumerate(subgoals.items()):

        targets.append(end_state_id)

        for start_state_id in subgoals.keys():
            if start_state_id == end_state_id:
                continue  # don't want self transitions

            try:
                path = nx.shortest_path(graph, start_state_id, end_state_id)
                if len(path) - 1 > max_length:
                    # too long - ignore
                    continue

                if len(path) == 2:
                    # too short - ignore
                    continue

                paths.append(path)
            except nx.NetworkXNoPath:
                pass
    return paths


def _get_edge_attributes(level, path, graph):
    weight = 1
    actions = list()
    length = 0
    observations = list()
    for i in range(len(path) - 1):
        edge = graph.edges[(path[i], path[i + 1])]
        a = edge['action']
        if isinstance(a, list):
            actions.extend(a)
        else:
            actions.append(a)
        if i == 0:
            observations.append(graph.nodes[path[i]]['obs'])
        observations.append(graph.nodes[path[i + 1]]['obs'])

        weight *= edge.get('weight', 1)
        length += 1
    return {'weight': weight, 'length': length, 'level': level, 'action': actions}

    # return {'weight': weight, 'length': length, 'level': level, 'action': actions, 'observations': observations}


def _recompute_connectivity(level, graph, reduced_graph, paths):
    reduced_graph.remove_edges_from(list(reduced_graph.edges()))

    for path in paths:
        reduced_graph.add_edge(path[0], path[-1])
        if not graph.has_edge(path[0], path[-1]):
            # not supporting multigraph. So only add edge if it doesn't already exist
            graph.add_edge(path[0], path[-1], **_get_edge_attributes(level, path, graph))
    # remove disconnected nodes
    isolates = list(nx.isolates(reduced_graph))
    reduced_graph.remove_nodes_from(isolates)


def compute_hierarchical_graph(graph: nx.DiGraph, max_length=4, subgoal_method='betweenness', verbose=False,
                               **kwargs):
    shrinking_graph = graph.copy()

    original_graph_per_level = graph.copy()
    graphs_per_level = {0: shrinking_graph.copy()}

    level = 1
    max_level = kwargs.get('max_level', np.inf)
    while level < max_level:
        paths = _identify_subgoals(shrinking_graph, max_length=max_length, subgoal_method=subgoal_method,
                                   verbose=verbose,
                                   **kwargs)

        _recompute_connectivity(level, original_graph_per_level, shrinking_graph, paths)
        # original_graph_per_level = graph.copy()
        graphs_per_level[level] = shrinking_graph.copy()

        level += 1
        if len(paths) <= 2:
            # can't get any smaller!
            break

    return original_graph_per_level
