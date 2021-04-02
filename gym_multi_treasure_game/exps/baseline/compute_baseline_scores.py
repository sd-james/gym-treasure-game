import traceback
from functools import partial

import networkx as nx
from tqdm import trange

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.exps.eval2 import evaluate_one_step, evaluate_n_step, \
    compute_histogram
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from s2s.env.s2s_env import View
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.utils import make_path, Recorder, range_without, exists, save, run_parallel, load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# matplotlib.use('TkAgg')


def try_build(save_dir, task, n_episodes, verbose=False):
    env = MockTreasureGame(task)

    domain, problem, info = build_transfer_model(env, None, None,
                                                 reload=True,
                                                 save_dir=save_dir,
                                                 n_jobs=16,
                                                 seed=None,
                                                 n_episodes=n_episodes,
                                                 options_per_episode=1000,
                                                 view=View.AGENT,
                                                 **CONFIG[task],
                                                 visualise=False,
                                                 save_data=False,
                                                 verbose=verbose)
    return env, domain, problem, info['n_samples'], info['copied_symbols'], info['copied_operators']


def extract_scores(recorder, task, n_exps, n_samples):
    dir = '/media/hdd/treasure_data'
    scores = list()
    samples = list()
    for exp in range(n_exps):
        score = list()
        sample = list()
        for n in range(1, 51):
            score.append(recorder.get_score(exp, task, n))
            sample.append(n_samples[(exp, task, n)])
        scores.append(score)
        samples.append(sample)
    return np.mean(samples, axis=0), np.mean(scores, axis=0), np.std(scores, axis=0)


def evaluate_models(dir, tasks, N, exact_length, n_exps=5):
    """
    Evaluate the prebuilt models for each task
    :param dir: the data directory
    :param tasks: the list of tasks
    :param N: the length of paths to be considered. -1 means all paths
    :param exact_length: whether we should consider only paths of length N, or <=N
    :param n_exps: the number of experiment runs
    """
    recorders = {i: Recorder() for i in tasks}
    all_stats = {i: Recorder() for i in tasks}
    for experiment in range(n_exps):
        for task in tasks:
            ground_truth = nx.read_gpickle('../data/ground_truth/graph_{}.pkl'.format(task))
            # draw(ground_truth, True)
            for n_episodes in range(1, 51):
                save_dir = make_path(dir, task, experiment, n_episodes)
                n_samples = -1
                try:
                    graph_path = make_path(save_dir, "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
                    assert exists(graph_path)
                    graph = nx.read_gpickle(graph_path)
                    # draw(graph, False)
                    if N == 1:
                        score = evaluate_one_step(ground_truth, graph)
                    else:
                        score, stats = evaluate_n_step(ground_truth, graph, n=N, exact_length=exact_length,
                                                       get_stats=True)
                        all_stats[task].record(experiment, task, n_episodes, stats)
                    n_samples = len(pd.read_pickle('{}/transition.pkl'.format(save_dir), compression='gzip'))
                    recorders[task].record(experiment, task, n_episodes, score)
                except Exception as e:
                    traceback.print_exc()
                    score = -1

                print(
                    "Experiment: {}\nTask: {}\nEpisodes: {}\nSamples: {}\nScore: {}\n"
                        .format(experiment, task, n_episodes, n_samples, score))
    return recorders, all_stats


def get_n_samples():
    samples = dict()
    tasks = range_without(1, 11)
    dir = '/media/hdd/treasure_data'
    n_exps = 5
    for experiment in trange(n_exps):
        for task in tasks:
            for n_episodes in trange(1, 51):
                save_dir = make_path(dir, task, experiment, n_episodes)
                n_samples = len(pd.read_pickle('{}/transition.pkl'.format(save_dir), compression='gzip'))
                samples[(experiment, task, n_episodes)] = n_samples
    return samples


def extract_stats(merged_stats, task, n_exps):
    ground_truth = nx.read_gpickle('info_graph_{}.pkl'.format(task))
    ground_stats = compute_histogram(ground_truth)
    pred_stats = merged_stats.get_score(0, task, 50)
    return ground_stats, pred_stats

    # for exp in range(n_exps):
    #     score = list()
    #     sample = list()
    #     for n in range(1, 51):
    #
    #
    #
    #         score.append(merged_stats.get_score(exp, task, n))
    #         sample.append(n_samples[(exp, task, n)])


if __name__ == '__main__':
    import warnings

    data_dir = '/media/hdd/treasure_data'
    save_dir = '../data'

    # get the number of samples in each dataset
    if not exists(make_path(save_dir, 'nsamples.pkl')):
        n_samples = get_n_samples()
        save(n_samples, 'nsamples.pkl')
    else:
        n_samples = load(make_path(save_dir, 'nsamples.pkl'))

    n_exps = 10
    exact_length = False
    N = -1
    n_jobs = 20
    tasks = range_without(1, 11)

    split = np.array_split(tasks, n_jobs)
    functions = [partial(evaluate_models, data_dir, split[i], N, exact_length, n_exps=n_exps) for i in
                 range(len(split))]
    result = run_parallel(functions)

    merged = Recorder()
    merged_stats = Recorder()
    for res, _ in result:
        for task, recorder in res.items():
            for a, b, c, d in recorder.iter():
                merged.record(a, b, c, d)
    for _, res in result:
        for task, recorder in res.items():
            for a, b, c, d in recorder.iter():
                merged_stats.record(a, b, c, d)

    save(merged, make_path(save_dir, 'baseline.pkl'))
    save(merged_stats, make_path(save_dir, 'baseline_stats.pkl'))

    for i, task in enumerate(tasks):
        samples, means, devs = extract_scores(merged, task, n_exps, n_samples)
        plt.plot(samples, means)
        plt.fill_between(samples, means - devs, means + devs, alpha=0.5)
    plt.show()

    # merged_stats = load('baseline_stats.pkl')
    # for i, task in enumerate(tasks):
    #     ground, predicted = extract_stats(merged_stats, task, n_exps)
    #
    #     plt.style.use('seaborn-deep')
    #     bins = np.arange(1, 50)
    #     plt.hist([ground, predicted], bins, label=['Actual', 'Predicted'])
    #     plt.legend(loc='upper right')
    #
    #     # plt.hist(ground, bins=)
    #     # plt.hist(predicted, bins=max(ground))
    #     plt.show()
    #
    # exit(0)
    #
