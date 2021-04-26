import random
import traceback
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.exps.baseline.generate_test_cases import _get_random_path, _extract_plan
from gym_multi_treasure_game.exps.eval2 import build_graph, evaluate_n_step, __get_pos
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.exps.evaluate_plans import evaluate_plans
from gym_multi_treasure_game.exps.graph_utils import merge, clean, clean_and_fit, extract_pairs, merge_and_clean
from gym_multi_treasure_game.exps.transfer.new_transfer import get_best
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.domain import Domain
from s2s.env.s2s_env import View
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.portable.transfer import extract, _unexpand_macro_operator
from s2s.utils import make_path, save, load, Recorder, now, range_without, exists, files_in_dir
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns


def get_seed(filename: str):
    return int(filename[filename.rindex('_') + 1: filename.rindex('.')])


def extract_ordered(recorder, task_order):
    data = defaultdict(list)
    task_count_to_task = dict()
    for experiment, A in recorder.items():
        for task, B in A.items():
            task_count = task_order[task]
            task_count_to_task[task_count] = task
            for n_episodes, (to_keep, graph) in B.items():
                data[task_count].append((n_episodes, to_keep, graph))
    data = dict(sorted(data.items()))
    new_data = dict()
    for task_count, values in data.items():
        new_data[task_count_to_task[task_count]] = values
    return new_data


def load_data(base_dir):
    task_orders = dict()

    for dir, file in files_in_dir(base_dir):
        if file.startswith('ntrans'):
            recorder, _ = load(make_path(dir, file))
            task_order = dict()
            for experiment, A in recorder.items():
                for (task_count, task), B in A.items():
                    task_order[task] = task_count
            task_orders[get_seed(file)] = task_order

    recorders = list()
    count = 0
    for dir, file in tqdm(files_in_dir(base_dir)):

        if 'stats_' in file:
            print('.', end='', flush=True)
            recorder = load(make_path(dir, file))
            recorder = extract_ordered(recorder, task_orders[get_seed(file)])
            recorders.append(recorder)

            count += 1
            # if count > 3:
            #     break

    return recorders


def get_edges_copied(edges, graph):
    if edges is None:
        return 0, 1

    nodes = set()
    for node in graph.nodes:
        if not isinstance(node, int):
            nodes.add(node)

    total = 0
    count = 0
    for u, v in graph.edges():
        total += 1
        if u in nodes and v in nodes:
            count += 1
    return count, total


def get_nodes_copied(edges, graph):
    if edges is None:
        return 0

    nodes = list()
    for node in graph.nodes:
        if not isinstance(node, int):
            nodes.append(node)
    return len(nodes)


def get_predicates_copied(edges, graph):
    if edges is None:
        return 0, 1

    nodes = defaultdict(list)
    for node in graph.nodes:
        if isinstance(node, int):
            nodes[-1].append(node)
        else:
            task = node[0: node.index(':')]
            nodes[task].append(node)

    predicates = defaultdict(set)
    for task, data in nodes.items():
        for node in data:
            preds = {x.name for x in graph.nodes[node]['predicates'] if len(x.mask) > 0}
            predicates[task] |= preds

    total = 0
    transferred = 0
    for task, syms in predicates.items():
        total += len(syms)
        if task != -1:
            transferred += len(syms)

    return transferred, total


def extract_edge_data(total_data):
    record = list()
    for data in total_data:
        for i, (task, values) in enumerate(data.items()):
            n_episodes, to_keep, graph = values[-1]
            edges, total_edges = get_edges_copied(to_keep, graph)
            record.append([i, edges, edges / total_edges, "Transfer"])
    return pd.DataFrame(record, columns=['Number of tasks', 'Number of edges', 'Proportion of edges', "Type"])


def extract_node_data(total_data):
    record = list()
    for data in total_data:
        for i, (task, values) in enumerate(data.items()):
            n_episodes, to_keep, graph = values[-1]
            nodes = get_nodes_copied(to_keep, graph)
            record.append([i, nodes, nodes / len(graph.nodes), "From previous tasks"])
            record.append([i, len(graph.nodes) - nodes, 1 - nodes / len(graph.nodes), "From current task"])
    return pd.DataFrame(record, columns=['Number of tasks', 'Number of nodes', 'Proportion of nodes', "Type"])


def extract_predicate_data(total_data):
    record = list()
    for data in total_data:
        for i, (task, values) in enumerate(data.items()):
            n_episodes, to_keep, graph = values[-1]
            predicates, all_predicates = get_predicates_copied(to_keep, graph)
            record.append([i, predicates, predicates / all_predicates, "From previous tasks"])
            record.append([i, predicates, 1 - predicates / all_predicates, "From current task"])
    return pd.DataFrame(record, columns=['Number of tasks', 'Number of predicates', 'Proportion of predicates', "Type"])


def plot_predicates(data):
    data = extract_predicate_data(data)
    #
    # save(data)
    # exit(0)
    sns.set(style="whitegrid")
    # sns.lineplot(x="Number of tasks", y='Number of predicates', hue="Type", data=data)
    # plt.show()
    # sns.barplot(x="day", y="total_bill", hue="sex", data=tips)
    sns.barplot(x="Number of tasks", y='Proportion of predicates', hue="Type", data=data)
    # sns.lineplot(x="Number of tasks", y='Proportion of predicates', hue="Type", data=data)
    plt.savefig('predicate_transfer.pdf')
    plt.show()


def plot_nodes(data):
    data = extract_node_data(data)
    # sns.lineplot(x="Number of tasks", y='Number of nodes', hue="Type", data=data)
    sns.histplot(x="Number of tasks", y='Number of nodes', hue="Type", data=data, multiple="stack")

    plt.show()
    # sns.lineplot(x="Number of tasks", y='Proportion of nodes', hue="Type", data=data)
    # plt.show()


def plot_edges(data):
    data = extract_edge_data(data)
    sns.lineplot(x="Number of tasks", y='Number of edges', hue="Type", data=data)
    plt.show()
    sns.lineplot(x="Number of tasks", y='Proportion of edges', hue="Type", data=data)
    plt.show()


if __name__ == '__main__':
    length = 3
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    # data = load()

    base_dir = '../data'

    data = load_data('../data/transfer_results')
    plot_predicates(data)
    exit(0)
    plot_nodes(data)
    plot_edges(data)


    exit(0)
