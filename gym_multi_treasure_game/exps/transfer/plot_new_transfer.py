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


def extract_ordered(recorder):
    data = defaultdict(list)
    task_order = dict()
    for experiment, A in recorder.items():
        for (task_count, task), B in A.items():
            task_order[task_count] = task
            for n_episodes, score in B.items():
                data[task_count].append((experiment, n_episodes, score))
    data = dict(sorted(data.items()))
    new_data = dict()
    for task_count, values in data.items():
        new_data[task_order[task_count]] = values
    return new_data


def _extract_baseline_episodes_required(baseline, samples=None):
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


def extract_episodes_required(baseline, total_data, samples=None):
    record = _extract_baseline_episodes_required(baseline, samples=samples)
    for data in total_data:
        for i, (task, values) in enumerate(data.items()):
            print('{} = {}'.format(i, task))
            if i == 0:
                continue
            for (experiment, n_episodes, score) in values:
                if score >= 1:
                    # record.append([i, baseline_ep, "No Transfer"])
                    if samples is not None:
                        n_episodes = samples[(experiment, task, n_episodes)]
                    record.append([i, n_episodes, "Transfer"])
                    # record.append([i, n_episodes, "Transfer"])

                    break
    col_name = 'Number of {}'.format('samples' if samples is not None else 'episodes')
    return pd.DataFrame(record, columns=['Number of tasks', col_name, "Type"])


def load_data(base_dir):
    recorders = list()
    for dir, file in files_in_dir(base_dir):
        if file.startswith('ntrans'):
            recorder, _ = load(make_path(dir, file))
            recorder = extract_ordered(recorder)
            recorders.append(recorder)
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

    X_AXIS_EPISODES = True

    length = 3
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    base_dir = '../data'
    baseline, _ = load(make_path(base_dir, '{}_ntransfer_results.pkl'.format(length)))

    if X_AXIS_EPISODES:
        samples = None
    else:
        samples = get_n_samples(base_dir)

    data = load_data('../data/transfer_results')
    data = extract_episodes_required(baseline, data, samples=samples)
    sns.lineplot(x="Number of tasks", y="Number of {}".format('episodes' if X_AXIS_EPISODES else 'samples'), hue="Type",
                 data=data)
    plt.show()

    exit(0)

    length = 3
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    base_dir = '../data'
    # baseline = load(make_path(base_dir, 'baseline.pkl'))

    baseline, _ = load(make_path(base_dir, '{}_ntransfer_results.pkl'.format(length)))

    test_cases = load('../data/{}_test_cases.pkl'.format(length))

    import warnings

    warnings.filterwarnings("ignore")

    recorder, transfer_recorder = load(make_path(base_dir, 'transfer_results/ntransfer_results_{}.pkl'.format(seed)))

    data = extract_ordered(recorder)

    data = extract_episodes_required(data)
    sns.lineplot(x="Number of tasks", y="Number of episodes", hue="Type", data=data)
    plt.show()

    exit(0)

    record = list()

    for i, (task, values) in enumerate(data.items()):
        for (experiment, n_episodes, score) in values:
            baseline_ep, baseline_score = get_best(baseline, experiment, task)

            no_transfer = baseline.get_score(experiment, task, n_episodes) / baseline_score
            if i == 6:
                record.append([n_episodes, no_transfer, 'No transfer'])
                record.append([n_episodes, score, '{} tasks seen'.format(i)])
                print('{}({}): {} vs {}'.format(task, n_episodes, score, no_transfer))

    data = pd.DataFrame(record, columns=['Number of episodes', 'Score', "Tasks Seen"])
    sns.lineplot(x="Number of episodes", y="Score", hue="Tasks Seen", data=data)
    plt.show()
    # for experiment, A in recorder.items():
    #     for (task_count, task), B in A.items():
    #         for n_episodes, score in B.items():
    #             print('{}-{}({}): {} vs {}'.format(task, task_count, n_episodes, score,
    #                                                baseline.get_score(experiment, task, n_episodes)))
    # recorder.record(experiment, (task_count, task), n_episodes, score)

    exit(0)
    recorder = Recorder()
    transfer_recorder = Recorder()
    all_stats = Recorder()

    dir = '/media/hdd/treasure_data'
    tasks = range_without(1, 11)
    np.random.shuffle(tasks)

    experiments = np.random.randint(0, 5, size=10)

    USE_HIERARCHY = False

    previous_graph = None
    classifiers = dict()

    for task_count, task in tqdm(enumerate(tasks)):
        experiment = get_valid_exp(dir, task)
        baseline_ep, baseline_score = get_best(baseline, experiment, task)
        ground_truth = nx.read_gpickle(make_path(base_dir, 'ground_truth', 'graph_{}.pkl'.format(task)))

        best_score = -np.inf
        best_domain = None

        # for n_episodes in trange(1, 51):
        original_graph = None
        for n_episodes in range(1, 51):
            save_dir = make_path(dir, task, experiment, n_episodes)

            graph_path = make_path(save_dir, "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
            assert exists(graph_path)
            graph = nx.read_gpickle(graph_path)
            original_graph = graph.copy()

            data = pd.read_pickle(make_path(save_dir, "transition.pkl"), compression='gzip')
            graph, clusterer, to_keep = merge(graph, previous_graph, data, classifiers, n_jobs=20)

            # draw(graph, False)
            # draw(ground_truth, True)

            # raw_score, stats = 0.1, None

            raw_score = evaluate_plans(test_cases[task], ground_truth, graph, clusterer, n_jobs=20)
            stats = None
            # raw_score, stats = evaluate_n_step(ground_truth, graph, get_stats=True)
            score = raw_score / baseline_score
            print(n_episodes, raw_score, score, baseline.get_score(experiment, task, n_episodes))

            all_stats.record(experiment, task, n_episodes, (to_keep, graph))

            recorder.record(0, (experiment, task_count, task), n_episodes, score)
            # print(
            #     "Time: {}\nExperiment: {}\nTask: {}({})\nTarget: {}-{}\nHierarchy?: {}\nEpisodes: {}\nSamples: {}\nScore: {}\nPredicates: {}/{}\n"
            #     "Operators: {}/{}\n"
            #         .format(now(), experiment, task, task_count, baseline_ep, baseline_score, USE_HIERARCHY, n_episodes,
            #                 n_samples, score, n_syms, len(previous_predicates), n_ops, len(previous_operators)))

            # if domain is not None:
            #     transfer_recorder.record(experiment, (task_count, task), (n_episodes, n_samples),
            #                              (n_syms, n_ops, len(domain.predicates),
            #                               len(domain.operators),
            #                               len(previous_operators)))
            #
            # if score > best_score:
            #     best_score = score
            #     if domain is not None:
            #         best_domain = domain
            #     else:
            #         best_domain = n_episodes

            if score >= 1:
                # assert raw_score >= baseline_score
                break

        if isinstance(best_domain, int):
            path = make_path(dir, task, experiment, best_domain)
            print(path)
            # try:
            #     _, best_domain, _, _, _, _ = try_build(path, task, best_domain,
            #                                            previous_predicates, previous_operators)
            # except:
            #     _, best_domain, _, _, _, _ = try_build(path, task, best_domain,
            #                                            previous_predicates, previous_operators)

        # previous_predicates, previous_operators = get_transferable_symbols(best_domain, previous_predicates,
        #                                                                    previous_operators)
        print("Merging...")
        time = now()
        classifiers = clean_and_fit(classifiers, task, original_graph)
        previous_graph = merge_and_clean(previous_graph, original_graph, task, classifiers)
        print('Merging took {} ms'.format(now() - time))

    save(recorder, )
    save((recorder, transfer_recorder), make_path(base_dir, 'ntransfer_results.pkl'))
    save(all_stats, make_path(base_dir, 'transfer_stats.pkl'))
