import random
import traceback
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import sklearn
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

POSITIONS = defaultdict(list)


def extract_ordered(recorder):
    global POSITIONS
    data = defaultdict(list)
    task_order = dict()
    for experiment, A in recorder.items():
        for (task_count, task), B in A.items():
            task_order[task_count] = task
            for n_episodes, score in B.items():
                data[task_count].append((experiment, n_episodes, score))
    data = dict(sorted(data.items()))
    new_data = OrderedDict()
    for task_count, values in data.items():
        new_data[task_order[task_count]] = values

    for i, task in task_order.items():
        POSITIONS[task].append(i)

    return new_data


def _extract_baseline_episodes_required(baseline, samples=None, cumulative=False):
    record = list()

    for experiment in range(10):
        for task in range(1, 11):
            try:
                baseline_ep, baseline_score = get_best(baseline, experiment, task)

                if samples is not None:
                    baseline_ep = samples[(experiment, task, baseline_ep)]

                record.append([0, baseline_ep, "Transfer"])
                for i in range(10):
                    record.append([i, baseline_ep, "No Transfer"])
            except:
                pass
    return record


def compute_area(data):

    if len(data) == 1:
        data = data + data

    return sklearn.metrics.auc(np.arange(len(data)), data)



def compute_ratio(no_transfer, transfer):
    assert len(no_transfer) == len(transfer)
    auc_transfer = compute_area(transfer)
    auc_no_transfer = compute_area(no_transfer)
    return (auc_transfer - auc_no_transfer) / auc_no_transfer


def extract_scores(baseline, total_data):
    record = list()

    for data in total_data:  # each experiment


        for i, (task, values) in enumerate(data.items()):
            transfer = list()
            no_transfer = list()

            if i == 0:
                continue

            for (experiment, n_episodes, score) in values:

                baseline_score = baseline.get_score(experiment, task, n_episodes)
                no_transfer.append(baseline_score)
                transfer.append(score)
                if score >= 1:
                    break
            record.append([i, compute_ratio(no_transfer, transfer)])
    return pd.DataFrame(record, columns=['Number of tasks', "Transfer ratio"])


def load_data(base_dir):
    recorders = list()
    for dir, file in files_in_dir(base_dir):
        if file.startswith('ntrans'):
            recorder, _ = load(make_path(dir, file))
            recorder = extract_ordered(recorder)
            recorders.append(recorder)
    print("Got {}".format(len(recorders)))
    return recorders


def get_n_samples(base_dir):
    path = make_path(base_dir, 'transfer_results', 'nsamples.pkl')
    if exists(path):
        return load(path)

    samples = dict()
    tasks = range_without(1, 11)
    dir = '/media/hdd/treasure_data'
    n_exps = 10
    for experiment in trange(n_exps):
        for task in tasks:
            for n_episodes in trange(1, 51):
                save_dir = make_path(dir, task, experiment, n_episodes)
                try:
                    n_samples = len(pd.read_pickle('{}/transition.pkl'.format(save_dir), compression='gzip'))
                    samples[(experiment, task, n_episodes)] = n_samples
                except:
                    pass
    save(samples, path)
    return samples


if __name__ == '__main__':
    length = 3
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    base_dir = '../data'
    baseline, _ = load(make_path(base_dir, '{}_ntransfer_results.pkl'.format(length)))

    data = load_data('../data/transfer_results')

    # for task, v in POSITIONS.items():
    #     plt.hist(v)
    #     plt.title(str(task))
    #     plt.show()
    #     plt.clf()
    #     print(task, np.mean(v))
    # exit(0)

    data = extract_scores(baseline, data)
    sns.set_theme(style="whitegrid")
    sns.boxplot(x="Number of tasks", y="Transfer ratio", data=data, fliersize=0)
    plt.show()
