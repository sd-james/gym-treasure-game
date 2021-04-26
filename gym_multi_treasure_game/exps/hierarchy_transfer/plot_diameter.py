import random
from collections import defaultdict
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from gym_multi_treasure_game.exps.graph_utils import merge, clean_and_fit, merge_and_clean
from gym_multi_treasure_game.exps.hierarchy_transfer.generate_hierarchy import compute_hierarchical_graph
from gym_multi_treasure_game.exps.transfer.new_transfer import draw
from gym_multi_treasure_game.exps.transfer.plot_new_transfer import load_data
from s2s.utils import make_path, save, now, exists, load, run_parallel
import matplotlib.pyplot as plt


def digraph_diameter(graph):
    sum = 0
    count = 0
    max_len = -np.inf
    for node in graph.nodes:
        paths = nx.single_source_dijkstra_path_length(graph, node, weight='cost')
        for x, path in paths.items():
            if x != node:
                sum += path - 1
                count += 1
                max_len = max(max_len, path-1)
    return sum / count, max_len

    return nx.average_shortest_path_length(graph, weight='cost')

    # paths = dict(nx.all_pairs_shortest_path(graph, cutoff=None))
    # max_path = -np.inf
    # for start, items in paths.items():
    #     for end, path in items.items():
    #         if start == end:
    #             continue
    #         max_path = max(max_path, len(path) - 1)
    # return max_path



def _my_diameter(G, full_paths):
    paths = dict(nx.all_pairs_shortest_path(G, cutoff=None))
    max_path = -np.inf
    # for start, items in paths.items():
    #     for end, path in items.items():
    #         if start == end:
    #             continue
    #         max_path = max(max_path, len(path) - 1)

    for start, items in full_paths.items():
        for end, path in items.items():
            if start == end:
                continue
            if start not in paths or end not in paths[start]:
                length = path
            else:
                length = len(paths[start][end])
            max_path = max(max_path, length - 1)

    return max_path


def diameter(graph, level):

    # replace
    for u, v, edge in graph.edges(data=True):
        if isinstance(u, int) and isinstance(v, int):
            # from current task
            # edge['cost'] = edge['length']
            if edge['level'] > level:
                # do not use
                edge['cost'] = edge['length']
            else:
                edge['cost'] = 1
        else:
            # from different task
            if edge['level'] > level:
                # do not use
                edge['cost'] = edge['length']
            else:
                edge['cost'] = 1

        graph.add_edge(u, v, **edge)

    # # remove levels above the one we're considering
    # edges = [(u, v, edge) for u, v, edge in graph.edges(data=True) if
    #          edge['level'] >= level and (isinstance(u, str) or isinstance(v, str))]
    # graph.remove_edges_from(edges)
    # # remove hierarchy from current task
    # edges = [(u, v, edge) for u, v, edge in graph.edges(data=True) if
    #          edge['level'] > 0 and (isinstance(u, int) and isinstance(v, int))]
    # graph.remove_edges_from(edges)
    return digraph_diameter(graph)
    return _my_diameter(graph, paths)


def get_stats(task_count, after_graph):
    # if current graph has no path, use above but pay full price!
    diameters = dict()
    for level in range(4):
        # print(level)
        diameters[level + 1] = diameter(after_graph.copy(), level)
    # for level in range(2, 5):
    #     if level == 1:
    #         diameters[level] = diameter(before_graph)
    #     else:
    #         diameters[level] = diameter(after_graph.copy(), level)

    return [[task_count, level, diameters[level][0], diameters[level][1]] for level, edges in diameters.items()]


def process(results):
    dir = '/media/hdd/treasure_data'
    task_count = 0
    records = list()
    for (after_graph, saved_hierarchy) in results:
        records.extend(get_stats(task_count, after_graph))
        task_count += 1
    return records


def process_parallel(indices):
    results = list()
    for i in tqdm(indices):
        path = '../data/hierarchy_results/hierarchy_{}.pkl'.format(i)
        if exists(path):
            records = load(path)[0]
            results.extend(process(records))
    return results


if __name__ == '__main__':
    #
    # data = pd.read_pickle('plot_data.pkl', compression='gzip')
    # g = sns.barplot(x="Number of tasks", y="Proportion of edges transferred", hue="Level", data=data)
    # g.set_yscale("log")
    #
    # # # plt.savefig("edges.pdf")
    # #
    # # g = sns.barplot(x="Number of tasks", y="Diameter", hue="Level", data=data)
    # # # plt.savefig("diameter.pdf")
    # #
    # plt.show()
    # #
    # exit(0)

    import warnings

    warnings.filterwarnings("ignore")

    length = 3
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # indices = np.array_split(np.arange(110), 20)
    # results = run_parallel([partial(process_parallel, x) for x in indices])
    # results = sum(results, [])
    #
    #
    # print("GOT {}".format(len(results)))
    # data = pd.DataFrame(results,
    #                     columns=['Number of tasks', "Level", "Average length", "Diameter"])
    #
    # pd.to_pickle(data, 'diameter.pkl')

    data = pd.read_pickle('diameter.pkl')

    sns.set(style="whitegrid")
    # sns.barplot(x="Number of tasks", y="Diameter", hue="Level", data=data, ci='sd')
    # plt.savefig('diameter_var.pdf')
    sns.barplot(x="Number of tasks", y="Average length", hue="Level", data=data)
    plt.savefig('average_ci.pdf')

    plt.show()
