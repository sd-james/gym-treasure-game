from gym_multi_treasure_game.exps.eval2 import build_graph


import traceback

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from s2s.env.s2s_env import View
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.utils import make_path, exists
import numpy as np
import networkx as nx


def try_build(save_dir, task, n_episodes, previous_predicates, previous_operators, build_hierarchy=False,
              verbose=False):
    env = MockTreasureGame(task)

    if len(previous_predicates) == 0:
        previous_predicates = None
    if len(previous_operators) == 0:
        previous_operators = None

    domain, problem, info = build_transfer_model(env, previous_predicates, previous_operators,
                                                 reload=True,
                                                 save_dir=save_dir,
                                                 n_jobs=23,
                                                 seed=None,
                                                 n_episodes=n_episodes,
                                                 options_per_episode=1000,
                                                 view=View.AGENT,
                                                 **CONFIG[task],
                                                 visualise=False,
                                                 save_data=False,
                                                 verbose=verbose)
    return env, domain, problem, info['n_samples'], info['copied_symbols'], info['copied_operators']


if __name__ == '__main__':

    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('experimentname')
    # parser.add_argument('task')
    # args = parser.parse_args()
    # experiment = int(args.experimentname)

    import warnings

    warnings.filterwarnings("ignore")

    dir = '/media/hdd/treasure_data'
    # tasks = range(int(args.task), int(args.task) + 1)
    tasks = range(1, 11)
    previous_predicates = list()
    previous_operators = list()
    for experiment in range(5):
        for task_count, task in enumerate(tasks):
            ground_truth = nx.read_gpickle('graph_{}.pkl'.format(task))
            best_score = -np.inf
            best_domain = None
            for n_episodes in range(1, 51):
                save_dir = make_path(dir, task, experiment, n_episodes)

                if exists(make_path(save_dir, "info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))):
                    continue

                try:
                    env, domain, problem, n_samples, n_syms, n_ops = try_build(save_dir, task, n_episodes,
                                                                               previous_predicates,
                                                                               previous_operators)
                    graph = build_graph(domain)
                    nx.write_gpickle(graph, make_path(save_dir, "info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes)))
                except:
                    traceback.print_exc()