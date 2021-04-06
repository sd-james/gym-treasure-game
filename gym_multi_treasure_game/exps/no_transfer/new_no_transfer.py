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
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.domain import Domain
from s2s.env.s2s_env import View
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.portable.transfer import extract, _unexpand_macro_operator
from s2s.utils import make_path, save, load, Recorder, now, range_without, exists
import numpy as np
import networkx as nx
import pandas as pd

if __name__ == '__main__':

    length = 4

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    base_dir = '../data'
    test_cases = load('../data/{}_test_cases.pkl'.format(length))

    import warnings

    warnings.filterwarnings("ignore")

    recorder = Recorder()
    transfer_recorder = Recorder()
    all_stats = Recorder()

    dir = '/media/hdd/treasure_data'
    tasks = range_without(1, 11)

    for task_count, task in tqdm(enumerate(tasks)):
        print("TASK {}".format(task))

        for experiment in range(10):
            print("EXPERIMENT {}".format(experiment))

            if not exists(make_path(dir, task, experiment, 49,
                                    "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, 49))):
                continue
            ground_truth = nx.read_gpickle(make_path(base_dir, 'ground_truth', 'graph_{}.pkl'.format(task)))
            for n_episodes in trange(1, 51):
                save_dir = make_path(dir, task, experiment, n_episodes)

                graph_path = make_path(save_dir,
                                       "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
                assert exists(graph_path)
                graph = nx.read_gpickle(graph_path)
                data = pd.read_pickle(make_path(save_dir, "transition.pkl"), compression='gzip')
                graph, clusterer = merge(graph, None, data, None, n_jobs=20)

                # draw(graph, False)
                # draw(ground_truth, True)

                # raw_score, stats = 0.1, None

                score = evaluate_plans(test_cases[task], ground_truth, graph, clusterer, n_jobs=20)
                print(n_episodes, score)
                stats = None

                all_stats.record(experiment, task, n_episodes, stats)
                recorder.record(experiment, task, n_episodes, score)
            # print(
            #     "Time: {}\nExperiment: {}\nTask: {}({})\nTarget: {}-{}\nHierarchy?: {}\nEpisodes: {}\nSamples: {}\nScore: {}\nPredicates: {}/{}\n"
            #     "Operators: {}/{}\n"
            #         .format(now(), experiment, task, task_count, baseline_ep, baseline_score, USE_HIERARCHY, n_episodes,
            #                 n_samples, score, n_syms, len(previous_predicates), n_ops, len(previous_operators)))

        save(recorder, )
        save((recorder, transfer_recorder), make_path(base_dir, '{}_ntransfer_results.pkl'.format(length)))
        save(all_stats, make_path(base_dir, '{}_transfer_stats.pkl'.format(length)))
