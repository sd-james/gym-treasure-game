import traceback

import networkx as nx
import numpy as np
import pandas as pd

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.eval2 import build_graph, evaluate_n_step
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from pyddl.pddl.domain import Domain
from s2s.env.s2s_env import View
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.portable.transfer import extract, _unexpand_macro_operator
from s2s.utils import make_path, save, load, Recorder, range_without, exists


def build_abstractions(domain):
    if domain is None:
        return domain
    tasks = discover_hddl_tasks(domain, verbose=True, draw=False, subgoal_method='voterank')
    operators = list()
    for task in tasks:
        for method in task.methods:
            operators.append(method.flatten())
    linked_ops = set()
    for operator in operators:
        assert 'operators' in operator.data
        chain = operator.data['operators']
        operator.data['operators'] = [x.data['parent'] for x in chain]
        linked_ops.add(_unexpand_macro_operator(operator))
    return linked_ops


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
                                                 n_jobs=16,
                                                 seed=None,
                                                 n_episodes=n_episodes,
                                                 options_per_episode=1000,
                                                 view=View.AGENT,
                                                 **CONFIG[task],
                                                 visualise=False,
                                                 save_data=False,
                                                 verbose=verbose)
    if build_hierarchy:
        methods = build_abstractions(domain)
        for op in methods:
            domain.add_linked_operator(op)
        # domain = build_abstractions(domain)
    return env, domain, problem, info['n_samples'], info['copied_symbols'], info['copied_operators']


def get_transferable_symbols(domain: Domain, previous_predicates, previous_operators):
    curr_preds, curr_ops = extract(domain)
    predicates = set(previous_predicates)
    predicates.update(curr_preds)
    operators = set(previous_operators)
    operators.update(curr_ops)
    return list(predicates), list(operators)


def scale(baseline, experiment, task, score):
    baseline = max([baseline.get_score(experiment, task, i) for i in range(1, 51)])
    # baseline = baseline.get_score(experiment, task, 50)
    return score / baseline


if __name__ == '__main__':

    baseline = load('baseline.pkl')

    THRESHOLD = np.inf  # 0.5
    import warnings

    warnings.filterwarnings("ignore")

    recorder = Recorder()
    all_stats = Recorder()

    dir = '/media/hdd/treasure_data'
    tasks = range_without(1, 11)
    USE_HIERARCHY = False

    for experiment in range(5):
        # np.random.shuffle(tasks)

        for task_count, task in enumerate(tasks):
            ground_truth = nx.read_gpickle('graph_{}.pkl'.format(task))

            best_score = -np.inf
            best_domain = None

            for n_episodes in range(1, 51):
                save_dir = make_path(dir, task, experiment, n_episodes)
                domain = None
                n_samples = len(pd.read_pickle('{}/transition.pkl'.format(save_dir), compression='gzip'))
                n_syms = 0
                n_ops = 0
                try:
                    graph_path = make_path(save_dir, "info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
                    assert exists(graph_path)
                    graph = nx.read_gpickle(graph_path)
                    score, stats = evaluate_n_step(ground_truth, graph, get_stats=True)
                    score = scale(baseline, experiment, task, score)
                    all_stats.record(experiment, task, n_episodes, stats)
                except Exception as e:
                    traceback.print_exc()
                    score = 0

                recorder.record(experiment, (task_count, task), n_episodes, score)
                print(
                    "Experiment: {}\nTask: {}\nHierarchy?: {}\nEpisodes: {}\nSamples: {}\nScore: {}\n"
                        .format(experiment, task, USE_HIERARCHY, n_episodes, n_samples, score, n_syms))

                if score > best_score:
                    best_score = score
                    best_domain = n_episodes

                if score >= THRESHOLD:
                    break

        save(recorder, 'no_transfer_results.pkl')
        save(all_stats, 'no_transfer_stats.pkl')
