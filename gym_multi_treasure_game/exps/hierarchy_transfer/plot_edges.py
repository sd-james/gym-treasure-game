import random
from collections import defaultdict
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import seaborn as sns
from gym_multi_treasure_game.exps.graph_utils import merge, clean_and_fit, merge_and_clean
from gym_multi_treasure_game.exps.hierarchy_transfer.generate_hierarchy import compute_hierarchical_graph
from gym_multi_treasure_game.exps.transfer.plot_new_transfer import load_data
from s2s.utils import make_path, save, now, exists, load, run_parallel
import matplotlib.pyplot as plt


def get_stats(task_count, after_graph, saved_hierarchy):
    levels = dict()
    for i in range(1, 5):
        levels[i] = list()

    for u, v, edge in after_graph.edges(data=True):
        if isinstance(u, str) or isinstance(v, str):
            levels[edge['level'] + 1].append(edge)

    for _, _, edge in saved_hierarchy:
        levels[edge['level'] + 1].append(edge)

    total = sum(len(edges) for _, edges in levels.items())

    return [[task_count, len(edges), len(edges) / total, "Level {}".format(level)] for level, edges in levels.items()]


def process(results):
    task_count = 0
    records = list()
    for (after_graph, saved_hierarchy) in results:
        if task_count != 0:
            records.extend(get_stats(task_count, after_graph, saved_hierarchy))
        task_count += 1
    return records


def process_parallel(indices):
    results = list()
    for i in indices:
        path = '../data/hierarchy_results/hierarchy_{}.pkl'.format(i)
        if exists(path):
            records = load(path)[0]
            results.extend(process(records))
    return results


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    length = 3
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # indices = np.array_split(np.arange(200), 20)
    # results = run_parallel([partial(process_parallel, x) for x in indices])
    # results = sum(results, [])
    #
    # # results = list()
    # # for i in trange(200):
    # #     path = '../data/hierarchy_results/hierarchy_{}.pkl'.format(i)
    # #     if exists(path):
    # #         records = load(path)[0]
    # #         results.extend(process(records))
    #
    #
    # print("GOT {}".format(len(results)))
    # data = pd.DataFrame(results,
    #                     columns=['Number of tasks seen', "Number of edges transferred", "Proportion of edges transferred",
    #                              "Level"])
    #
    # pd.to_pickle(data, 'edges.pkl')
    data = pd.read_pickle('edges.pkl')

    sns.set(style="whitegrid")
    g = sns.barplot(x="Number of tasks seen", y="Proportion of edges transferred", hue="Level", data=data, ci="sd")
    # g.legend(loc='center left', bbox_to_anchor=(1.01, 1), ncol=1)
    plt.legend(bbox_to_anchor=(1, 1))
    g.set_yscale("log")
    plt.savefig("edges.pdf", bbox_inches='tight')
    plt.show()
