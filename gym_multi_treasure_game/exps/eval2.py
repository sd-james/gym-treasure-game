import itertools
import warnings
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gym_multi_treasure_game.envs.pca.base_pca import PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.utils import plan_parallel
from pyddl.pddl.domain import Domain
from pyddl.pddl.operator import Operator
from pyddl.pddl.predicate import Predicate
from s2s.hierarchy.discover_hddl_methods import _generate_mdp, _generate_full_mdp
from s2s.hierarchy.network import Node, Path
from s2s.hierarchy.option_discovery import _get_path
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import save, run_parallel, load

import matplotlib

# MAPPER = dict()

MAPPER = load("/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/transfer/MAPPER.pkl")
KNN = load('/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/transfer/knn.pkl')

#
# pca2 = PCA(PCA_INVENTORY)
# pca2.load('/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/envs/pca/models/dropped_key_pca_inventory.dat')
# matplotlib.use('TkAgg')
def _combine(state, inv):

    # global MAPPER
    global KNN

    has_key = 0
    has_gold = 0

    temp = tuple(np.around(inv.astype(float), 3))
    ret = KNN.predict([temp])[0]
    if ret == 1:
        has_key = 1
    elif ret == 2:
        has_gold = 1

    # if temp in MAPPER:
    #     has_key, has_gold = MAPPER[temp]
    #     print("FOUND")
    # else:
    #     print(temp)
    #     img = pca2.unflatten(pca2.uncompress_(inv)).astype(float)
    #     plt.imshow(img)
    #     plt.show()
    #
    #     i = int(input("HAS KEY/GOLD?"))
    #     if i == 0:
    #         has_key = 0
    #         has_gold = 0
    #     elif i == 1:
    #         has_key = 1
    #         has_gold = 0
    #     else:
    #         has_key = 0
    #         has_gold = 1
    #     MAPPER[temp] = (has_key, has_gold)
    #     save(MAPPER, "/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/transfer/MAPPER.pkl")
    #     plt.clf()


    # if inv is not None:
    #     m = np.mean(inv)
    #     if -1.6 <= m <= -1.5:
    #         pass
    #     elif 0.7 <= m <= 0.8:
    #         if has_key != 1:
    #             warnings.warn("WRONG!A")
    #         has_key = 1
    #     elif 3.7 <= m <= 3.8:
    #         if has_key != 1:
    #             warnings.warn("WRONG!B")
    #         has_key = 1
    #     elif -0.7 <= m <= -0.5:
    #         if has_key != 0 or has_gold != 1:
    #             warnings.warn("WRONG!C")
    #         has_key = 0
    #         has_gold = 1
    #     elif 1.5 <= m <= 1.6:
    #         if has_key != 0 or has_gold != 1:
    #             warnings.warn("WRONG!D")
    #         has_key = 0
    #         has_gold = 1
    #     elif m >= 1.35:
    #         if has_key != 0 or has_gold != 1:
    #             warnings.warn("WRONG!E")
    #         has_key = 0
    #         has_gold = 1
    #     else:
    #         raise ValueError

    return np.hstack((state, [has_key, has_gold]))


def _score_reachable(predicted, actual, **kwargs):
    n_jobs = kwargs.get('n_jobs', 1)

    candidates = list()
    for start_node in actual.nodes:
        for end_node in actual.nodes:
            if start_node == end_node or not nx.has_path(actual, start_node, end_node):
                continue
            candidates.append((start_node, end_node))

    if 'max' in kwargs:
        N = kwargs['max']
        candidates = list(map(tuple, np.array(candidates)[np.random.choice(len(candidates), N, replace=False)]))

    splits = np.array_split(candidates, n_jobs)
    functions = [partial(_count_shortest, predicted, actual, splits[i]) for i in range(n_jobs)]
    results = run_parallel(functions)
    find = 0
    count = 0
    for f, t in results:
        find += f
        count += t
    return find / count


def _dist(a, b):
    m = min(len(a), len(b))
    a = a[0:m]
    b = b[0:m]
    for i in range(len(a)):
        if abs(a[i] - b[i]) > 0.09:
            return np.inf
    return np.linalg.norm(a - b)


def _is_close(x, y, tol=0.000001):
    return abs(x - y) < tol


def _find_node(state, graph, keep_all=False):
    if len(state.shape) == 2:
        state = state.squeeze()
    if keep_all:
        best_node = list()
    else:
        best_node = None
    best_dist = np.inf
    for node in graph.nodes:
        s = graph.nodes[node]['state']
        if len(s.shape) == 2:
            s = s.squeeze()
        s = _combine(s, graph.nodes[node]['obs'][1])
        dist = _dist(s, state)

        if keep_all:

            if dist < best_dist:
                best_dist = dist
                best_node.append(node)
            elif not np.isinf(dist) and _is_close(best_dist, dist):
                best_node.append(node)
        elif dist < best_dist:
            best_dist = dist
            best_node = node
    return best_node


def _count_shortest(predicted, actual, candidates):
    find = 0

    for start_node, end_node in candidates:
        source = _find_node(actual.nodes[start_node]['state'], predicted)
        if source is None:
            continue
        target = _find_node(actual.nodes[end_node]['state'], predicted)
        if target is None or not nx.has_path(predicted, source, target):
            continue
        true_path = nx.shortest_path(actual, start_node, end_node)

        pred_path = nx.shortest_path(predicted, source, target)
        if len(pred_path) == len(true_path):
            find += 1

        # if target is not None and nx.has_path(predicted, source, target):
        #     find += 1
    return find, len(candidates)


def evaluate_other(ground_truth, graph, **kwargs):
    return _score_reachable(graph, ground_truth, **kwargs), graph


def _is_match(shortest, starts, ends, options=None):
    for x, y in itertools.product(starts, ends):
        if x in shortest and y in shortest[x]:

            if options is None:
                return True

            return True
    return False


def _get_idx(x, path):
    return [i for i, y in enumerate(path) if x in y]


def _is_match_n_step(mapping, shortest, starts, ends, true_path, use_hierarchy=False):
    for x, y in itertools.product(starts, ends):
        if not x in shortest or y not in shortest[x]:
            continue

        pred_path = shortest[x][y]
        if len(pred_path) > len(true_path):
            continue
        if use_hierarchy and len(pred_path) < len(true_path):

            to_match = [mapping[x] for x in true_path]
            candidates = [_get_idx(x, to_match) for x in pred_path]

            if not any(len(x) == 0 for x in candidates):
                return True  # TODO approximation!

            # if is_increasing(canidates):
            #     # if there is a path
            #     return True

        else:
            if len(pred_path) != len(true_path):
                continue
            match = True
            for true, pred in zip(true_path, pred_path):
                if pred not in mapping[true]:
                    match = False
                    break
            if match:
                return True

    return False


def evaluate_one_step(ground_truth, graph):
    paths = dict(nx.all_pairs_shortest_path(ground_truth, cutoff=1))
    mapping = _find_mapping(graph, ground_truth)
    shortest = dict(nx.all_pairs_shortest_path(graph), cutoff=1)
    count = 0
    find = 0
    for start, items in paths.items():
        A = mapping[start]
        for end, path in items.items():
            if start == end:
                continue
            B = mapping[end]
            count += 1
            if _is_match(shortest, A, B):
                find += 1
    return find / count


def compute_histogram(graph):
    paths = dict(nx.all_pairs_shortest_path(graph, cutoff=None))
    A = list()
    for start, items in paths.items():
        for end, path in items.items():
            if start == end:
                continue
            A.append(len(path))
    return A


def evaluate_n_step(ground_truth, graph, n=-1, exact_length=False, get_stats=False, use_hierarchy=False):
    if n == -1:
        n = None
    paths = dict(nx.all_pairs_shortest_path(ground_truth, cutoff=n))
    mapping = _find_mapping(graph, ground_truth)
    shortest = dict(nx.all_pairs_shortest_path(graph), cutoff=n)
    count = 0
    find = 0
    stats_B = list()
    for start, items in paths.items():
        A = mapping[start]
        for end, path in items.items():
            if start == end:
                continue
            if exact_length:
                if n is not None and len(path) - 1 != n:
                    continue
            B = mapping[end]
            count += 1

            if _is_match_n_step(mapping, shortest, A, B, path, use_hierarchy=use_hierarchy):
                find += 1
                if get_stats:
                    stats_B.append(len(path))
    if get_stats:
        return find / count, stats_B
    return find / count


def _find_mapping(graph, actual):
    actual_to_predicted = defaultdict(list)
    for node in actual.nodes:
        curr = _find_node(actual.nodes[node]['state'], graph, keep_all=True)
        actual_to_predicted[node] = curr
    return actual_to_predicted


def sample_paths(graph, N):
    candidates = list()
    for start_node in graph.nodes:
        for end_node in graph.nodes:
            if start_node == end_node or not nx.has_path(graph, start_node, end_node):
                continue
            candidates.append((start_node, end_node))
    candidates = list(map(tuple, np.array(candidates)[np.random.choice(len(candidates), N, replace=False)]))

    paths = list()
    for start, end in candidates:
        paths.append(nx.shortest_path(graph, start, end))
    return paths


def _op_length(operator):
    if 'options' in operator.data:
        return len(operator.data['options'])
    return 1


def _op_level(operator):
    level = 0
    while 'operators' in operator.data:
        level += 1
        operator = operator.data['operators'][0]
    return level


def _sample(state):
    ground_state = [None] * 2
    for pred in state.predicates:
        data = np.mean(pred.sample(100), axis=0)
        for i, m in enumerate(pred.mask):
            ground_state[m] = data[i]
    return ground_state


def __get_pos(pos, obs):
    mean = _combine(pos, obs)
    pos = np.array([mean[0], 1 - mean[1]])
    if mean[2] == 1:
        pos += [0.01, -0.02]
    if len(mean) >= 4 and mean[3] == 1:
        pos += [0.03, 0.03]
    if len(mean) >= 5 and mean[4] == 1:
        pos += [-0.02, -0.02]
    return pos


def build_graph(domain):
    portable_symbols = [x for x in domain.predicates if x != Predicate.not_failed() and 'psymbol' not in x.name]
    problem_symbols = [x for x in domain.predicates if x != Predicate.not_failed() and 'psymbol' in x.name]

    states, transitions = _generate_full_mdp(portable_symbols, problem_symbols, domain.linked_operators)

    graph = nx.DiGraph()
    for start, edges in transitions.items():
        for edge in edges:
            end = edge.end_node
            graph.add_edge(start.id, end.id, weight=edge.prob, length=_op_length(edge.action), action=edge.action,
                           level=_op_level(edge.action))

    temp = dict()
    for predicate in domain.predicates:
        if isinstance(predicate, _ProblemProposition):
            link = int(predicate.name[predicate.name.index('_') + 1:])
            temp[link] = predicate
    positions = dict()
    for state in states:
        symbol = temp[state.link]
        problem_data = symbol.sample(1)[0]
        obs_data = _sample(state)
        positions[state.id] = __get_pos(problem_data, obs_data[1])
        graph.nodes[state.id]['state'] = problem_data
        graph.nodes[state.id]['obs'] = obs_data
    return graph
