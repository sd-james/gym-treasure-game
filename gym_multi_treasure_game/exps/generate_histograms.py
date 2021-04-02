import itertools
from collections import defaultdict, deque, OrderedDict
from typing import List, Set

import networkx as nx
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from gym_multi_treasure_game.exps.transfer.new_transfer import get_best, draw
from pyddl.hddl.hddl_task import HDDLTask
from pyddl.pddl.predicate import Predicate
from pyddl.pddl.problem import Problem
from s2s.hierarchy.abstract_state import AbstractState
from s2s.hierarchy.network import Node, Edge
from s2s.hierarchy.option_discovery import compute_hierarchical_options, _find_elbow
from s2s.pddl.domain import Domain
from s2s.pddl.pddl_operator import Operator
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import show, make_path, load, exists, range_without

KNN = load('/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/data/knn.pkl')


def identify_subgoals(graph: nx.DiGraph, max_length=4, subgoal_method='betweenness', verbose=False,
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

                # TODO we just need to use the start and end only right? Must check
                paths.append((path[0], path[-1]))
            except nx.NetworkXNoPath:
                pass
    return paths


def recompute_connectivity(graph, reduced_graph, paths):
    reduced_graph.remove_edges_from(list(reduced_graph.edges()))

    for path in paths:
        reduced_graph.add_edge(path[0], path[-1])
        graph.add_edge(path[0], path[-1])
    # remove disconnected nodes
    isolates = list(nx.isolates(reduced_graph))
    reduced_graph.remove_nodes_from(isolates)


def compute_hierarchical_options(graph: nx.DiGraph, max_length=4, subgoal_method='betweenness', verbose=False,
                                 **kwargs):
    shrinking_graph = graph.copy()

    original_graph_per_level = {0: graph.copy()}
    graphs_per_level = {0: shrinking_graph.copy()}

    level = 1
    max_level = kwargs.get('max_level', np.inf)
    while level < max_level:
        paths = identify_subgoals(shrinking_graph, max_length=max_length, subgoal_method=subgoal_method,
                                  verbose=verbose,
                                  **kwargs)

        recompute_connectivity(graph, shrinking_graph, paths)
        original_graph_per_level[level] = graph.copy()
        graphs_per_level[level] = shrinking_graph.copy()

        level += 1
        if len(paths) <= 2:
            # can't get any smaller!
            break

    return original_graph_per_level, graphs_per_level


def show_histos(graphs, title):
    def _get_count(G):
        paths = dict(nx.all_pairs_shortest_path(G, cutoff=None))
        A = list()
        for start, items in paths.items():
            for end, path in items.items():
                if start == end:
                    continue
                A.append(len(path) - 1)
        return A

    data = list()
    to_skip = {3, 5}
    to_skip.clear()
    for level, graph in graphs.items():
        if level + 1 in to_skip:
            continue
        for x in _get_count(graph):
            data.append([x, "Level {}".format(level + 1)])
    data = pd.DataFrame(data, columns=['Optimal path length', 'Max abstraction level'])
    sns.set(style="whitegrid")
    sns.displot(data, x="Optimal path length", hue="Max abstraction level", binwidth=1, element="step")
    # plt.savefig('xxx.pdf')
    plt.title(title)
    plt.show()
    # exit(0)


def load_graph(task, experiment=None):
    if experiment is None:
        while True:
            experiment = np.random.randint(10)
            try:
                base_dir = 'data'
                baseline = load(make_path(base_dir, 'baseline.pkl'))
                baseline_ep, baseline_score = get_best(baseline, experiment, task)
                dir = '/media/hdd/treasure_data'
                save_dir = make_path(dir, task, experiment, baseline_ep)
                graph_path = make_path(save_dir,
                                       "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, baseline_ep))
                if not exists(graph_path):
                    continue
                graph = nx.read_gpickle(graph_path)
                return graph
            except:
                continue
    else:
        try:
            base_dir = 'data'
            baseline = load(make_path(base_dir, 'baseline.pkl'))
            baseline_ep, baseline_score = get_best(baseline, experiment, task)
            dir = '/media/hdd/treasure_data'
            save_dir = make_path(dir, task, experiment, baseline_ep)
            graph_path = make_path(save_dir, "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, baseline_ep))
            if not exists(graph_path):
                return None
            graph = nx.read_gpickle(graph_path)
            return graph
        except:
            return None


def is_goal(node):
    """
    Convert the inventory image into  a simple vector and combine with the state info
    """
    state = node['state']
    if state[2] != 1:
        return False

    inv = node['obs'][1]

    global KNN

    temp = tuple(np.around(inv.astype(float), 3))
    ret = KNN.predict([temp])[0]
    return ret == 2


def is_start(node):
    """
    Convert the inventory image into  a simple vector and combine with the state info
    """
    inv = node['obs'][1]
    global KNN
    temp = tuple(np.around(inv.astype(float), 3))
    ret = KNN.predict([temp])[0]
    if ret != 0:
        return False
    state = node['state']
    if state[2] == 1:
        return False

    predicates = {x.name for x in node['predicates']}
    if predicates != {'symbol_0', 'symbol_1'}:
        return False
    x, y = state[0], state[1]
    return x < 0.25 and y < 0.15


def relabel(graph, task):
    labels = {node: "{}: {}".format(task, node) for node in graph.nodes}
    return nx.relabel_nodes(graph, labels)


def link(graphs):
    new_graphs = list()
    for i, graph in enumerate(graphs):
        new_graphs.append(relabel(graph, i))

    edges = list()
    for i in range(0, len(new_graphs) - 1):
        A = new_graphs[i]
        B = new_graphs[i + 1]
        ends = [node for node in A.nodes if is_goal(A.nodes[node])]
        starts = [node for node in B.nodes if is_start(B.nodes[node])]
        for u in ends:
            for v in starts:
                edges.append((u, v))
    graph = nx.compose_all(new_graphs)
    for u, v in edges:
        graph.add_edge(u, v, weight=1)
    return graph


if __name__ == '__main__':

    graphs = [load_graph(i) for i in range_without(1, 11, 3)]
    graphs = [load_graph(i) for i in range_without(1, 3)]
    C = link(graphs)
    original_graph_per_level, graphs_per_level = compute_hierarchical_options(C, max_length=4,
                                                                              reduce_graph=2,
                                                                              subgoal_method='voterank',
                                                                              verbose=True)
    show_histos(original_graph_per_level, "Task {}")
    exit(0)
    A = load_graph(1, 0)
    # draw(A, False)
    B = load_graph(2, 0)
    # draw(B, False)
    C = link([A, B])
    original_graph_per_level, graphs_per_level = compute_hierarchical_options(C, max_length=4,
                                                                              reduce_graph=3,
                                                                              subgoal_method='voterank',
                                                                              verbose=True)
    show_histos(original_graph_per_level, "Task {}")
    exit(0)

    for task in range(1, 11):
        for exp in range(10):
            graph = load_graph(task, exp)
            if graph is None:
                continue
            print("IN {}".format(exp))
            draw(graph, False)
            original_graph_per_level, graphs_per_level = compute_hierarchical_options(graph, max_length=4,
                                                                                      reduce_graph=3,
                                                                                      subgoal_method='voterank',
                                                                                      verbose=True)
            show_histos(original_graph_per_level, "Task {}, Exp {}".format(task, exp))
