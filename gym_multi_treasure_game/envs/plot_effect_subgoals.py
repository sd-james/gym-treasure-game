# plot the effect of various subgoal ID techniques
from collections import defaultdict
from typing import List, Dict, Tuple

import networkx as nx
import seaborn as sns
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from pyddl.hddl.hddl_method import HDDLMethod
from pyddl.hddl.hddl_task import HDDLTask
from s2s.hierarchy.discover_hddl_methods import _generate_mdp
from s2s.hierarchy.network import Edge, Node
from s2s.hierarchy.option_discovery import compute_options, construct_abstract_options
from s2s.pddl.domain import Domain
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import make_path, load
from s2s.utils import show

TOTAL_GRAPH = None


def _recompute_connectivity(graph, hddl_methods: Dict[Tuple[Node, Node], List[HDDLMethod]], paths):
    reduced_graph = nx.create_empty_copy(graph)
    global TOTAL_GRAPH

    for path in paths:
        reduced_graph.add_edge(path[0], path[-1])
        TOTAL_GRAPH.add_edge(path[0], path[-1])
    # remove disconnected nodes
    isolates = list(nx.isolates(reduced_graph))
    reduced_graph.remove_nodes_from(isolates)

    transition = defaultdict(list)
    for (start_node, end_node), methods in hddl_methods.items():
        for method in methods:
            transition[start_node].append(Edge(start_node, method, end_node, 1, 0))
    return reduced_graph, transition


def compute_hierarchical_options(graph: nx.DiGraph, states, transition: Dict[Node, List[Edge]], max_length=4,
                                 subgoal_method='betweenness', verbose=False, **kwargs):
    # levels
    # nodes per level
    # options per level
    # diameter per level

    node_stats = dict()
    option_stats = dict()
    diameter_stats = dict()

    original = graph.copy()

    global TOTAL_GRAPH
    TOTAL_GRAPH = graph.copy()

    level = 0

    node_stats[level] = len(graph.nodes)
    diameter_stats[level] = nx.diameter(TOTAL_GRAPH.to_undirected())

    max_level = kwargs.get('max_level', np.inf)
    while level < max_level:
        level += 1
        options, paths = compute_options(graph, states, transition, max_length=max_length,
                                         subgoal_method=subgoal_method,
                                         verbose=verbose, **kwargs)

        abstract_options = construct_abstract_options(options)

        option_stats[level] = len(abstract_options)

        method_edges = defaultdict(list)

        options, paths = compute_options(graph, states, transition, max_length=max_length,
                                         subgoal_method=subgoal_method,
                                         verbose=verbose, **kwargs)

        abstract_options = construct_abstract_options(options)
        method_edges = defaultdict(list)

        for option in abstract_options:
            task = HDDLTask('{}-Level-{}'.format(option.name[option.name.rindex('!') + 1:], level))
            for path in option.paths:
                method = HDDLMethod(task)
                for edge in path.walk():
                    method.add_subtask(edge.action)

                method_edges[(path.start_node(), path.end_node())].append(method)
                assert len(method) > 1
                task.add_method(method)

        graph, transition = _recompute_connectivity(graph, method_edges, paths)
        node_stats[level] = len(graph.nodes)
        diameter_stats[level] = nx.diameter(TOTAL_GRAPH.to_undirected())

        if len(paths) <= 2:
            # can't get any smaller!
            break

    return level, node_stats, option_stats, diameter_stats


def discover_hddl_tasks(domain: Domain, verbose=False, **kwargs) -> List[HDDLTask]:
    show('Generating abstract MDP...', verbose)

    states, transitions = _generate_mdp(kwargs['problem'].init, domain.linked_operators, kwargs['initial_link'])

    A = np.zeros((len(states), len(states)))
    for start, edges in transitions.items():
        for edge in edges:
            end = edge.end_node
            A[start.id, end.id] = 1

    graph = nx.from_numpy_array(A, create_using=nx.DiGraph)

    stats = compute_hierarchical_options(graph, states, transitions, max_length=4,
                                         verbose=verbose, **kwargs)
    return stats


def get_stats(method, reduce):
    TASK = 1
    save_dir = '../data.bak/{}'.format(TASK)
    domain = load(make_path(save_dir, 'linked_domain.pkl'))
    problem = load(make_path(save_dir, 'linked_problem.pkl'))

    prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    start_link = int(prop.name[prop.name.index('_') + 1:])

    if reduce > -1:
        return discover_hddl_tasks(domain, verbose=True, draw=False,
                                   subgoal_method=method,
                                   problem=problem,
                                   initial_link=start_link,
                                   reduce_graph=reduce)
    else:
        return discover_hddl_tasks(domain, verbose=True, draw=False,
                                   subgoal_method=method,
                                   problem=problem,
                                   initial_link=start_link)




def plot1(method):
    x_axis = list(range(2, 6))
    if method == 'betweenness':
        x_axis.append(-1)

    data = pd.DataFrame(columns=['Graph reduction', 'Nodes', "Level"])

    for i, reduce in enumerate(x_axis):
        levels, node_stats, option_stats, diameter_stats = get_stats(method, reduce)
        for l, n in node_stats.items():
            if l == 0:
                continue
            if reduce == -1:
                reduce = "Adaptive"
            data.loc[len(data)] = [reduce, n, l + 1]
    sns.set(style="whitegrid")
    sns.catplot(x="Graph reduction", y="Nodes", hue="Level", kind="bar", data=data)
    # plt.show()
    plt.savefig('{}-nodes.pdf'.format(method))

    data = pd.DataFrame(columns=['Graph reduction', 'Options', "Level"])

    for i, reduce in enumerate(x_axis):
        levels, node_stats, option_stats, diameter_stats = get_stats(method, reduce)
        for l, n in option_stats.items():
            if l == 0:
                continue
            if reduce == -1:
                reduce = "Adaptive"
            data.loc[len(data)] = [reduce, n, l + 1]
    sns.set(style="whitegrid")
    sns.catplot(x="Graph reduction", y="Options", hue="Level", kind="bar", data=data)
    # plt.show()
    plt.savefig('{}-options.pdf'.format(method))


    data = pd.DataFrame(columns=['Graph reduction', 'Diameter', "Level"])

    for i, reduce in enumerate(x_axis):
        levels, node_stats, option_stats, diameter_stats = get_stats(method, reduce)
        for l, n in diameter_stats.items():
            if l == 0:
                continue
            if reduce == -1:
                reduce = "Adaptive"
            data.loc[len(data)] = [reduce, n, l + 1]
    sns.set(style="whitegrid")
    sns.catplot(x="Graph reduction", y="Diameter", hue="Level", kind="bar", data=data)
    # plt.show()
    plt.savefig('{}-diameter.pdf'.format(method))



if __name__ == '__main__':

    plot1('betweenness')
    plot1('voterank')
    exit(0)

    x = list()
    y = list()

    x_axis = list(range(2, 6))

    data = pd.DataFrame(columns=['ReduceBy', 'Nodes', "Level"])

    for i, reduce in enumerate(x_axis):
        levels, node_stats, option_stats, diameter_stats = get_stats('voterank', reduce)
        for l, n in node_stats.items():
            if l == 0:
                continue
            data.loc[len(data)] = [reduce, n, l + 1]
    sns.set(style="whitegrid")
    sns.catplot(x="ReduceBy", y="Nodes", hue="Level", kind="bar", data=data)
    plt.show()
    exit(0)

    x = list()
    y = list()
    for reduce in range(2, 10):
        levels, node_stats, option_stats, diameter_stats = get_stats('voterank', reduce)
        x.append(reduce)
        y.append(levels)

    df = pd.DataFrame({
        "reduce": x,
        "column1": y,
    })

    #
    # df = pd.DataFrame({
    #     "date": ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"],
    #     "column1": [555, 525, 532, 585],
    #     "column2": [50, 48, 49, 51]
    # })

    sns.set(style="whitegrid")

    sns.lineplot(data=df.column1, color="g")
    plt.show()

    # sns.lineplot(data=df.column1, color="g")
    # ax2 = plt.twinx()
    # sns.lineplot(data=df.column2, color="b", ax=ax2)
    # plt.show()
