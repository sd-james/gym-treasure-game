import random
import traceback

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.exps.eval2 import build_graph, evaluate_n_step, __get_pos
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.exps.graph_utils import merge, clean, fit_classifiers
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

if __name__ == '__main__':

    seed=42
    random.seed(seed)
    np.random.seed(seed)

    base_dir = '../data'
    baseline = load(make_path(base_dir, 'baseline.pkl'))

    import warnings

    warnings.filterwarnings("ignore")

    recorder = Recorder()
    transfer_recorder = Recorder()
    all_stats = Recorder()

    dir = '/media/hdd/treasure_data'
    tasks = range_without(1, 11)
    np.random.shuffle(tasks)

    experiments = np.random.randint(0, 5, size=10)

    USE_HIERARCHY = False


    previous_graphs = list()
    classifiers = dict()

    for task_count, (task, experiment) in tqdm(enumerate(zip(tasks, experiments))):
        baseline_ep, baseline_score = get_best(baseline, experiment, task)
        ground_truth = nx.read_gpickle(make_path(base_dir, 'ground_truth', 'graph_{}.pkl'.format(task)))

        best_score = -np.inf
        best_domain = None

        # for n_episodes in trange(1, 51):
        original_graph = None
        for n_episodes in trange(baseline_ep, 51):
            save_dir = make_path(dir, task, experiment, n_episodes)

            graph_path = make_path(save_dir, "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
            assert exists(graph_path)
            graph = nx.read_gpickle(graph_path)
            original_graph = graph.copy()
            if len(previous_graphs) > 0:
                data = pd.read_pickle(make_path(save_dir, "transition.pkl"), compression='gzip')
                graph = merge(graph, previous_graphs, data, classifiers)

            # draw(graph, False)
            # draw(ground_truth, True)

            raw_score, stats = 0.6, None
            # raw_score, stats = evaluate_n_step(ground_truth, graph, get_stats=True)
            score = raw_score / baseline_score
            all_stats.record(experiment, task, n_episodes, stats)


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
                assert raw_score >= baseline_score
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

        previous_graphs.append((task, clean(original_graph)))
        classifiers = fit_classifiers(classifiers, task, original_graph)

    save(recorder, )
    save((recorder, transfer_recorder), make_path(base_dir, 'ntransfer_results.pkl'))
    save(all_stats, make_path(base_dir, 'transfer_stats.pkl'))
