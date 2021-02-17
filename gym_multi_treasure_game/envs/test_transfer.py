import sys

from memory_profiler import profile

sys.path.append('/home/steve/PycharmProjects/pyddl')
sys.path.append('/home/steve/PycharmProjects/hierarchical-skills-to-symbols')
sys.path.append('/home/steve/PycharmProjects/gym-multi-treasure-game')


import traceback

from tqdm import tqdm, trange

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.eval2 import build_graph, evaluate_n_step
from gym_multi_treasure_game.envs.evaluate import evaluate_manually, evaluate_with_network, validate, \
    evaluate_similarity
from gym_multi_treasure_game.envs.mock_env import MockTreasureGame
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.domain import Domain
from s2s.env.s2s_env import View
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.build_model_transfer import build_transfer_model
from s2s.portable.problem_symbols import _ProblemProposition
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
    baseline = baseline.get_score(experiment, task, 50)
    return score / baseline


if __name__ == '__main__':

    baseline = load('baseline.pkl')

    THRESHOLD = np.inf  # 0.5
    import warnings

    warnings.filterwarnings("ignore")

    recorder = Recorder()
    transfer_recorder = Recorder()
    all_stats = Recorder()

    dir = '/media/hdd/treasure_data'
    tasks = range_without(1, 5)
    USE_HIERARCHY = False

    for experiment in range(1):
        np.random.shuffle(tasks)
        previous_predicates = list()
        previous_operators = list()

        for task_count, task in tqdm(enumerate(tasks)):
            ground_truth = nx.read_gpickle('graph_{}.pkl'.format(task))

            best_score = -np.inf
            best_domain = None

            for n_episodes in trange(1, 51):
                for op in previous_operators:
                    op.clear()
                save_dir = make_path(dir, task, experiment, n_episodes)

                if task_count > 0 or USE_HIERARCHY:
                    env, domain, problem, n_samples, n_syms, n_ops = try_build(save_dir, task, n_episodes,
                                                                               previous_predicates,
                                                                               previous_operators,
                                                                               build_hierarchy=USE_HIERARCHY)
                else:
                    domain = None
                    n_samples = len(pd.read_pickle('{}/transition.pkl'.format(save_dir), compression='gzip'))
                    n_syms = 0
                    n_ops = 0
                try:

                    if domain is None:
                        graph_path = make_path(save_dir, "info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
                        assert exists(graph_path)
                        graph = nx.read_gpickle(graph_path)
                    else:
                        graph = build_graph(domain)
                    score, stats = evaluate_n_step(ground_truth, graph, get_stats=True)
                    score = scale(baseline, experiment, task, score)
                    all_stats.record(experiment, task, n_episodes, stats)

                    # score, graph = evaluate_similarity(ground_truth, domain, draw=False, n_jobs=16)
                    # nx.write_gpickle(graph, make_path(save_dir, "graph_{}_{}_{}.pkl".format(experiment, task, n_episodes)))
                except Exception as e:
                    traceback.print_exc()
                    score = 0
                    # found = False

                recorder.record(experiment, (task_count, task), n_episodes, score)
                print(
                    "Time: {}\nExperiment: {}\nTask: {}({})\nHierarchy?: {}\nEpisodes: {}\nSamples: {}\nScore: {}\nPredicates: {}/{}\n"
                    "Operators: {}/{}\n"
                        .format(now(), experiment, task, task_count, USE_HIERARCHY, n_episodes, n_samples, score,
                                n_syms,
                                len(previous_predicates), n_ops, len(previous_operators)))

                if domain is not None:
                    transfer_recorder.record(experiment, (task_count, task), (n_episodes, n_samples),
                                             (n_syms, n_ops, len(domain.predicates),
                                              len(domain.operators),
                                              len(previous_operators)))

                if score > best_score:
                    best_score = score
                    if domain is not None:
                        best_domain = domain
                    else:
                        best_domain = n_episodes

                if score >= THRESHOLD:
                    break

            if isinstance(best_domain, int):
                path = make_path(dir, task, experiment, best_domain)
                print(path)
                try:
                    _, best_domain, _, _, _, _ = try_build(path, task, best_domain,
                                                           previous_predicates, previous_operators)
                except:
                    _, best_domain, _, _, _, _ = try_build(path, task, best_domain,
                                                           previous_predicates, previous_operators)

            previous_predicates, previous_operators = get_transferable_symbols(best_domain, previous_predicates,
                                                                               previous_operators)
        save((recorder, transfer_recorder), 'transfer_results.pkl')
        save(all_stats, 'transfer_stats.pkl')
