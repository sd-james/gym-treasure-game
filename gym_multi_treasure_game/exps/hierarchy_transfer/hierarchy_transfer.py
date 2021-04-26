import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from gym_multi_treasure_game.exps.eval2 import __get_pos
from gym_multi_treasure_game.exps.evaluate_plans import evaluate_plans
from gym_multi_treasure_game.exps.graph_utils import merge, clean_and_fit, merge_and_clean
from gym_multi_treasure_game.exps.hierarchy_transfer.generate_hierarchy import compute_hierarchical_graph
from pyddl.pddl.domain import Domain
from s2s.env.s2s_env import View
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.portable.transfer import extract, _unexpand_macro_operator
from s2s.utils import make_path, save, load, Recorder, now, range_without, exists


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


# @profile
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


def get_best(baseline, experiment, task):
    best = -1
    best_idx = -1
    for i in range(1, 51):
        score = baseline.get_score(experiment, task, i)
        if score > best:
            best = score
            best_idx = i

    return best_idx, best


def draw(graph, ground_truth, show=True):
    positions = dict()
    if ground_truth:
        for node in graph.nodes:
            positions[node] = graph.nodes[node]['pos']
    else:
        for node in graph.nodes:
            problem_data = graph.nodes[node]['state']
            obs_data = graph.nodes[node]['obs']
            positions[node] = __get_pos(problem_data, obs_data[1])
    nx.draw(graph, pos=positions)
    if show:
        plt.show()


def get_valid_exp(dir, task):
    experiments = range_without(0, 10)
    np.random.shuffle(experiments)

    for experiment in experiments:
        if exists(make_path(dir, task, experiment, 49,
                            "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, 49))):
            return experiment
    return None


if __name__ == '__main__':

    length = 3
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    base_dir = '../data'
    # baseline = load(make_path(base_dir, 'baseline.pkl'))

    baseline, _ = load(make_path(base_dir, '{}_ntransfer_results.pkl'.format(length)))

    test_cases = load('../data/3_test_cases.pkl')

    import warnings

    warnings.filterwarnings("ignore")

    recorder = Recorder()
    transfer_recorder = Recorder()
    all_stats = Recorder()

    dir = '/media/hdd/treasure_data'
    tasks = range_without(1, 11)
    np.random.shuffle(tasks)

    print(tasks)

    experiments = np.random.randint(0, 5, size=10)

    USE_HIERARCHY = False

    previous_graph = None
    classifiers = dict()

    start = now()

    for task_count, task in tqdm(enumerate(tasks), desc="Tasks"):

        print("TIME: {}".format(now() - start))

        experiment = get_valid_exp(dir, task)

        print("DOING {} {}".format(experiment, task))

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

            graph = compute_hierarchical_graph(graph, max_length=4, reduce_graph=3, subgoal_method='voterank')

            original_graph = graph.copy()

            data = pd.read_pickle(make_path(save_dir, "transition.pkl"), compression='gzip')
            graph, clusterer, to_keep, saved_hierarchy = merge(graph, previous_graph, data, classifiers, n_jobs=20)

            # draw(graph, False)
            # draw(ground_truth, True)

            # raw_score, stats = 0.1, None

            # draw(graph, False)
            # draw(nx.read_gpickle(graph_path), False)
            # print(evaluate_plans(test_cases[task], ground_truth, nx.read_gpickle(graph_path), clusterer, n_jobs=1))
            # print()
            # print(evaluate_plans(test_cases[task], ground_truth, graph, clusterer, n_jobs=1))

            # exit(0)
            # print(evaluate_plans(test_cases[task], ground_truth, nx.read_gpickle(graph_path), clusterer, n_jobs=20))

            raw_score = evaluate_plans(test_cases[task], ground_truth, graph, clusterer, n_jobs=20)
            stats = None
            # raw_score, stats = evaluate_n_step(ground_truth, graph, get_stats=True)
            score = raw_score / baseline_score
            print(n_episodes, raw_score, score, baseline.get_score(experiment, task, n_episodes))

            all_stats.record(experiment, task, n_episodes, (to_keep, graph, saved_hierarchy))

            recorder.record(experiment, (task_count, task), n_episodes, score)
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
    save((recorder, transfer_recorder), make_path(base_dir, 'hierarchy_transfer_results_{}.pkl'.format(seed)))
    save(all_stats, make_path(base_dir, 'hierarchy_transfer_stats_{}.pkl'.format(seed)))
