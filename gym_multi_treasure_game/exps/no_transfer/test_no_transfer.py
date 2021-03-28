import traceback

import networkx as nx
import numpy as np
import pandas as pd

from gym_multi_treasure_game.exps.eval2 import evaluate_n_step
from gym_multi_treasure_game.exps.eval3 import new_evaluate
from s2s.utils import make_path, save, load, Recorder, range_without, exists


def scale(baseline, experiment, task, score):
    baseline = max([baseline.get_score(experiment, task, i) for i in range(1, 51)])
    # baseline = baseline.get_score(experiment, task, 50)
    return score / baseline


def get_best(baseline, experiment, task):
    best = -1
    best_idx = -1
    for i in range(1, 51):
        score = baseline.get_score(experiment, task, i)
        if score > best:
            best = score
            best_idx = i

    return best_idx, best


if __name__ == '__main__':

    base_dir = '../data'
    baseline = load(make_path(base_dir, 'baseline.pkl'))

    import warnings

    # warnings.filterwarnings("ignore")

    recorder = Recorder()
    all_stats = Recorder()

    dir = '/media/hdd/treasure_data'
    tasks = range_without(1, 11)

    for experiment in range(5):
        # np.random.shuffle(tasks)

        for task_count, task in enumerate(tasks):

            baseline_ep, baseline_score = get_best(baseline, experiment, task)

            ground_truth = nx.read_gpickle(make_path(base_dir, 'ground_truth', 'graph_{}.pkl'.format(task)))

            best_score = -np.inf
            best_domain = None

            for n_episodes in range(1, 51):
                data_dir = make_path(dir, task, experiment, n_episodes)
                domain = None
                n_samples = len(pd.read_pickle('{}/transition.pkl'.format(data_dir), compression='gzip'))
                n_syms = 0
                n_ops = 0
                try:
                    graph_path = make_path(data_dir, "info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
                    assert exists(graph_path)
                    graph = nx.read_gpickle(graph_path)

                    raw_score = new_evaluate(ground_truth, graph, get_stats=False)
                    continue

                    raw_score, stats = evaluate_n_step(ground_truth, graph, get_stats=True)
                    score = raw_score / baseline_score
                    # score = scale(baseline, experiment, task, raw_score)
                    all_stats.record(experiment, task, n_episodes, stats)
                except Exception as e:
                    traceback.print_exc()
                    score = 0
                    raw_score = 0

                recorder.record(experiment, (task_count, task), n_episodes, score)
                print(
                    "Experiment: {}\nTask: {}\nEpisodes: {}\nSamples: {}\nScore: {}\n"
                        .format(experiment, task, n_episodes, n_samples, score, n_syms))

                if score > best_score:
                    best_score = score
                    best_domain = n_episodes

                if score >= 1:
                    assert raw_score >= baseline_score
                    assert baseline_ep == n_episodes
                    break

        save(recorder, make_path(base_dir, 'no_transfer_results.pkl'))
        save(all_stats, make_path(base_dir, 'no_transfer_stats.pkl'))
