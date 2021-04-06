import itertools
import traceback
from collections import defaultdict, ChainMap
from functools import partial
from typing import Set

import networkx as nx
import numpy as np
from tqdm import tqdm

from s2s.estimators.oc_svc import OCSupportVectorClassifier
from s2s.pddl.pddl import Predicate
from s2s.portable.quick_cluster import QuickCluster
from s2s.utils import load, run_parallel, now

KNN = load('/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/data/knn.pkl')


def extract_options(edge):
    if edge['level'] == 0:
        return [edge['action']]

    raise NotImplementedError


def multiple_shortest_path_edges(graph, source, target):
    """
    Return the edges along the shortest path from source to target, or None if no path exists
    """
    if not nx.has_path(graph, source, target):
        return None
    plans = list()
    prev_size = -1
    for path in nx.shortest_simple_paths(graph, source, target):
        path_graph = nx.path_graph(path)
        if prev_size > -1 and len(path_graph) > prev_size:
            break
        prev_size = len(path_graph)
        # Read attributes from each edge
        plan = list()
        for ea in path_graph.edges():
            edge = graph.edges[ea[0], ea[1]]
            plan.append(edge)
        plans.append(plan)
    return plans


def shortest_path_edges(graph, source, target):
    """
    Return the edges along the shortest path from source to target, or None if no path exists
    """
    if not nx.has_path(graph, source, target):
        return None
    path = nx.shortest_path(graph, source, target)
    path_graph = nx.path_graph(path)
    # Read attributes from each edge
    plan = list()
    for ea in path_graph.edges():
        edge = graph.edges[ea[0], ea[1]]
        plan.append(edge)
    return plan


def _combine(state, inv):
    """
    Convert the inventory image into  a simple vector and combine with the state info
    """
    global KNN

    has_key = 0
    has_gold = 0

    temp = tuple(np.around(inv.astype(float), 3))
    ret = KNN.predict([temp])[0]
    if ret == 1:
        has_key = 1
    elif ret == 2:
        has_gold = 1
    return np.hstack((state, [has_key, has_gold]))


def find_nodes(state, graph):
    """
    Return the nodes in the graph that are likely representative of the given state
    """

    def _dist(a, b):
        m = min(len(a), len(b))
        a = a[0:m]
        b = b[0:m]
        for i in range(len(a)):
            if abs(a[i] - b[i]) > 0.1:
                return np.inf
        return np.linalg.norm(a - b)

    if len(state.shape) == 2:
        state = state.squeeze()
    candidates = list()
    best_dist = np.inf
    for node in graph.nodes:
        s = graph.nodes[node]['state']
        if len(s.shape) == 2:
            s = s.squeeze()
        s = _combine(s, graph.nodes[node]['obs'][1])
        dist = _dist(s, state)

        if dist < best_dist:
            best_dist = dist
            candidates.append(node)
        elif not np.isinf(dist) and abs(best_dist - dist) < 0.000001:
            candidates.append(node)
    return candidates


def _iter(transition_data):
    option = transition_data['option']
    start = transition_data['agent_state']
    start2 = transition_data['state']
    end = transition_data['next_agent_state']
    end2 = transition_data['next_state']
    return [(o, x, y, a, b) for o, x, y, a, b in zip(option, start, end, start2, end2)]


def _is_similar(node, other_node, prev_task, classifiers):
    A = dict()
    B = dict()

    for predicate in node['predicates']:
        if len(predicate.mask) == 0:
            continue
        A[tuple(predicate.mask)] = predicate
    for predicate in other_node['predicates']:
        if len(predicate.mask) == 0:
            continue
        B[tuple(predicate.mask)] = predicate

    for mask, predicate in A.items():

        if len(mask) == 0:
            continue

        if mask not in B:
            return False
        other_predicate = B[mask]
        classifier = classifiers[(prev_task, other_predicate)]
        if classifier is None:
            return False

        data = predicate.sample(100)
        prob = classifier.probability(data, use_mask=False)
        if prob < 0.9:
            return False
    return True


def is_similar_predicate(predicate, other_predicate, classifier):
    """
    Determine if two predicates are similar
    :param predicate: a predicate
    :param other_predicate: another predicate
    :param classifier: a classifier for the second predicate
    """
    if predicate.mask != other_predicate.mask:
        return False
    if classifier is None:
        return False
    data = predicate.sample(100)
    prob = classifier.probability(data, use_mask=False)
    return prob > 0.05


def find_predicate_mapping(current_graph, previous_graph, previous_task, classifiers):
    """
    Find matching predicates between two graphs
    :param current_graph: the current graph
    :param previous_graph: an older graph
    :param previous_task:
    :param classifiers:
    :return:
    """
    current_predicates = extract_predicates(current_graph)
    prev_predicates = extract_predicates(previous_graph)
    new_to_old = defaultdict(list)
    old_to_new = defaultdict(list)
    for previous_predicate in prev_predicates:
        for current in current_predicates:
            if is_similar_predicate(current, previous_predicate, classifiers[(previous_task, previous_predicate)]):
                new_to_old[current].append(previous_predicate)
                old_to_new[previous_predicate].append(current)
    return new_to_old, old_to_new


def is_match(state, next_state, start_node, end_node):
    def similar(x, y):
        for a, b in zip(x, y):
            if np.linalg.norm(a - b, np.inf) > 1:
                return False
        return True

    # TODO use OC SVM with predicates
    return similar(state, start_node['obs']) and similar(next_state, end_node['obs'])


def extract_predicates(graph: nx.Graph) -> Set[Predicate]:
    """
    Get all the predicates from all the nodes in teh graph
    :param graph:
    :return: a set of predicates
    """
    all_predicates = set()
    for n in graph.nodes:
        all_predicates |= set(graph.nodes[n]['predicates'])
    return all_predicates


def find_similar_node(node, duplicate_predicates, current_graph):
    predicates = set()
    for predicate in node['predicates']:
        if len(predicate.mask) == 0:
            continue
        if predicate not in duplicate_predicates:
            return False, None
        predicates.add(duplicate_predicates[predicate])

    for node in current_graph.nodes:
        preds = {x for x in current_graph.nodes[node]['predicates'] if len(x.mask) > 0}
        if preds == predicates:
            return True, node
    return False, None

    # clusterer = QuickCluster(env.n_dims(other_view), kwargs.get('linking_threshold', 0.15))
    # for _, row in transition_data.iterrows():
    #     state, next_state = row[get_column_by_view('state', {'view': other_view})], row[
    #         get_column_by_view('next_state', {'view': other_view})]
    #     clusterer.add(state)
    #     clusterer.add(next_state)


def _refers_to(state, classifiers):
    if classifiers is None:
        return False
    if len(classifiers) != len(state):
        return False
    for i, s in enumerate(state):
        classifier = classifiers[i]
        if classifier is None:
            return False
        if classifier.probability(s, use_mask=False) < 0.1:
            return False
    return True


def refers_to(state, next_state, u, v, classifiers):
    return _refers_to(state, classifiers[u]) and _refers_to(next_state, classifiers[v])


class Hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self))


def find_similar(node, graph, old_to_new):
    """
    Find similar nodes in a new graph
    :param node: an old node
    :param graph: the new graph
    :param old_to_new: a mapping from old to new predicates
    :return:
    """
    matching_nodes = list()
    new_predicates = [(old_to_new[predicate]) for predicate in node['predicates']]
    if any(len(x) == 0 for x in new_predicates):
        return matching_nodes
    for n in graph.nodes:
        other_node = graph.nodes[n]
        for x in itertools.product(new_predicates):
            if set(other_node['predicates'] == set(x)):
                matching_nodes.append(n)
    return matching_nodes


def get_edges_to_keep(graph, classifiers, transition_data, clusterer):
    seen = set()
    to_keep = defaultdict(list)


    for option, d, dprime, s, sprime in _iter(transition_data):
        start_link = clusterer.get(s, index_only=True)
        end_link = clusterer.get(sprime, index_only=True)
        memo = dict()
        for u, v, a in graph.edges(data=True):
            if (u, v, start_link, end_link) in seen:
                continue

            if a['action'] != option:
                continue

            start_refers = memo.get(u, _refers_to(d, classifiers[u]))
            memo[u] = start_refers
            if not start_refers:
                continue
            end_refers = memo.get(v, _refers_to(dprime, classifiers[v]))
            memo[v] = end_refers
            if not end_refers:
                continue
            # if refers_to(d, dprime, u, v, classifiers):
            seen.add((u, v, start_link, end_link))
            to_keep[(u, v, Hashabledict(a))].append((s, sprime))

    return to_keep


def compute_mapping(current_graph, classifiers, to_keep):
    mapping = dict()
    for (u, v, edge), states in tqdm(to_keep):
        if u not in mapping:
            mapping[u] = find_similar_nodes(u, current_graph, classifiers)
        if v not in mapping:
            mapping[v] = find_similar_nodes(u, current_graph, classifiers)
    return mapping

def merge(current_graph, previous_graph, transition_data, classifiers, n_jobs):
    # assume current graph already linked
    # use transition data to find nodes to transfer in.
    # link up with problem specific data

    clusterer = None
    for _, d, dprime, s, sprime in _iter(transition_data):
        if clusterer is None:
            clusterer = QuickCluster(len(s), 0.1)
        clusterer.add(s)
        clusterer.add(sprime)
    to_keep = None
    if previous_graph is not None:
        print("Finding edges...")
        time = now()
        splits = np.array_split(transition_data, n_jobs)
        functions = [partial(get_edges_to_keep, previous_graph, classifiers, data, clusterer) for data in splits]
        result = run_parallel(functions)
        to_keep = dict(ChainMap(*result))
        print("Finding {} edges took {} ms".format(len(to_keep), now() - time))

        print("Finding mapping...")
        time = now()
        splits = np.array_split(list(to_keep.items()), n_jobs)
        functions = [partial(compute_mapping, current_graph, classifiers, data) for data in splits]
        result = run_parallel(functions)
        mapping = dict(ChainMap(*result))
        print("Computing mapping took {} ms".format(now() - time))

        # mapping = dict()
        #
        # for (u, v, edge), states in tqdm(to_keep.items()):
        #     if u not in mapping:
        #         mapping[u] = find_similar_nodes(u, current_graph, classifiers)
        #     if v not in mapping:
        #         mapping[v] = find_similar_nodes(u, current_graph, classifiers)
        print("Integrating to graph...")
        time = now()
        added = set()
        for (u, v, edge), states in tqdm(to_keep.items()):
            for s, sprime in states:
                start_link = clusterer.get(s, index_only=True)
                if (u, start_link) in added:
                    start_nodes = ["{}:{}".format(u, start_link)]
                else:
                    start_nodes = mapping[u]
                    start_nodes = [x for x in start_nodes if 'state' in current_graph.nodes[x] and
                                   clusterer.get(current_graph.nodes[x]['state'], index_only=True) == start_link]
                    if len(start_nodes) == 0:
                        # add to graph
                        added.add((u, start_link))
                        start_nodes = ["{}:{}".format(u, start_link)]
                        current_graph.add_node("{}:{}".format(u, start_link), **previous_graph.nodes[u])
                end_link = clusterer.get(sprime, index_only=True)
                if (v, end_link) in added:
                    end_nodes = ["{}:{}".format(v, end_link)]
                else:
                    end_nodes = mapping[v]
                    end_nodes = [x for x in end_nodes if 'state' in current_graph.nodes[x] and
                                 clusterer.get(current_graph.nodes[x]['state'], index_only=True) == end_link]
                    if len(end_nodes) == 0:
                        # add to graph
                        added.add((v, end_link))
                        end_nodes = ["{}:{}".format(v, end_link)]
                        current_graph.add_node("{}:{}".format(v, end_link), **previous_graph.nodes[v])
                for a in start_nodes:
                    for b in end_nodes:
                        current_graph.add_edge(a, b, **edge)
        print("Integrating took {} ms".format(now() - time))

    return current_graph, clusterer, to_keep
    # current_predicates = extract_predicates(current_graph)
    # prev_predicates = extract_predicates(previous_graph)
    # new_to_old = defaultdict(list)
    # old_to_new = defaultdict(list)
    # for previous_predicate in prev_predicates:
    #     for current in current_predicates:
    #         if is_similar_predicate(current, previous_predicate, classifiers[(previous_task, previous_predicate)]):
    #             new_to_old[current].append(previous_predicate)
    #             old_to_new[previous_predicate].append(current)

    # for the ones that will be added, see if they're already in the current graph!

    # new_to_old = defaultdict(list)
    # old_to_new = defaultdict(list)

    # newly_added = set()
    #
    # for (u, v, edge), (s, sprime) in to_keep.items():
    #     edge = dict(edge)
    #
    #     if previous_graph[u] in newly_added:
    #         start_nodes = [u]
    #     else:
    #         start_nodes = find_similar(previous_graph[u], current_graph, old_to_new)
    #
    #     if previous_graph[v] in newly_added:
    #         end_nodes = [v]
    #     else:
    #         end_nodes = find_similar(previous_graph[v], current_graph, old_to_new)
    #
    #     if len(start_nodes) == 0:
    #         start_nodes = [u]
    #         newly_added.add(u)
    #         current_graph.add_node(u, previous_graph[u])
    #     if len(end_nodes) == 0:
    #         end_nodes = [v]
    #         newly_added.add(v)
    #         current_graph.add_node(v, previous_graph[v])
    #
    #     for start in start_nodes:
    #         for end in end_nodes:
    #             current_graph.add_edge(start, end, **edge)
    #
    #     if u in current_graph.nodes and v in current_graph.nodes:
    #         pass
    #
    #     start_idx = clusterer.get(s, index_only=True)
    #     end_idx = clusterer.get(sprime, index_only=True)
    #
    # # ignore these ones!
    # # current_predicates = extract_predicates(current_graph)
    # # prev_predicates = extract_predicates(previous_graph)
    # #
    # # new_to_old = defaultdict(list)
    # # old_to_new = defaultdict(list)
    # # for previous_predicate in prev_predicates:
    # #     for current in current_predicates:
    # #         if is_similar_predicate(current, previous_predicate, classifiers[(previous_task, previous_predicate)]):
    # #             new_to_old[current].append(previous_predicate)
    # #             old_to_new[previous_predicate].append(current)
    #
    # # Add old edges to new graph if applicable
    # # to_link =
    # # for u, v, a in previous_graph.edges(data=True):
    # #
    # #     start_nodes = find_duplicate_nodes(u)
    # #     end_nodes = find_duplicate_nodes(v)
    # #
    # #     if len(start_nodes) == 0 and len(end_nodes) == 0:
    # #
    # #         current_graph.add_node(u, **previous_graph[u])
    # #         current_graph.add_node(v, **previous_graph[v])
    #
    # # we now have a mapping from previous to new predicates.
    # # Go through node-edge-nodes and integrate into current graph, but replace node with current node if exact match
    # old_node_to_new_entry = dict()  # map old node id to id in new graph
    # new_nodes = set()
    # for u, v, a in previous_graph.edges(data=True):
    #
    #     match_found, start_node = find_similar_node(previous_graph.nodes[u], duplicate_predicates, current_graph)
    #     if not match_found:
    #         # add new node to graph
    #         if u in old_node_to_new_entry:
    #             start_node = old_node_to_new_entry[u]
    #         else:
    #             new_id = len(current_graph.nodes)
    #             old_node_to_new_entry[u] = new_id
    #             start_node = new_id
    #             new_nodes.add(new_id)
    #             current_graph.add_node(start_node, **previous_graph.nodes[u])
    #
    #     match_found, end_node = find_similar_node(previous_graph.nodes[v], duplicate_predicates, current_graph)
    #     if not match_found:
    #         # add new node to graph
    #         if v in old_node_to_new_entry:
    #             end_node = old_node_to_new_entry[v]
    #         else:
    #             new_id = len(current_graph.nodes)
    #             old_node_to_new_entry[v] = new_id
    #             end_node = new_id
    #             new_nodes.add(new_id)
    #             current_graph.add_node(end_node, **previous_graph.nodes[v])
    #     current_graph.add_edge(start_node, end_node, **a)
    #
    # # use transition data to link
    # found_nodes = set()
    # for d, dprime, s, sprime in tqdm(_iter(transition_data)):
    #
    #     # find matching start/end nodes
    #     for u, v, a in current_graph.edges(data=True):
    #         if is_match(d, dprime, current_graph.nodes[u], current_graph.nodes[v]):
    #             found_nodes.add(u)
    #             found_nodes.add(v)
    #
    # for node in current_graph.nodes:
    #     if node not in found_nodes:
    #         current_graph.remove_node(node)
    # current_graph.remove_nodes_from(list(nx.isolates(current_graph)))
    #
    # # use transition data to find nodes to transfer in.
    # matches = list()
    # for d, dprime, s, sprime in _iter(transition_data):
    #
    #     for u, v, a in previous_graph.edges(data=True):
    #         if u in exists or v in exists:
    #             # ignore
    #             continue
    #
    #         if is_match(d, dprime, previous_graph.nodes[u], previous_graph.nodes[v]):
    #             matches.append((u, v, a, s, sprime))
    #
    # # integrate matches into existing graph!
    # for start_node, end_node, edge, s, s_prime in matches:
    #     pass


def clean(graph):
    for node in graph.nodes:
        del graph.nodes[node]['state']
    return graph


def extract_pairs(task_id: int, graph: nx.Graph):
    """
    Extract start-end nodes with single edge only.
    :param graph:
    :return:
    """

    new_graph = nx.DiGraph()

    predicates = lambda node: tuple(
        sorted([predicate.name for predicate in node['predicates'] if len(predicate.mask) > 0]))

    visited = dict()
    for u, v, a in graph.edges(data=True):
        start = graph.nodes[u]
        end = graph.nodes[v]
        s_preds = predicates(start)
        e_preds = predicates(end)

        start = visited.get(s_preds, start)
        end = visited.get(e_preds, end)

        # u = "{}:{}".format(task_id, s_preds)
        # v = "{}:{}".format(task_id, e_preds)

        if u not in new_graph:
            new_graph.add_node(u, **start, task=task_id)
        if v not in new_graph:
            new_graph.add_node(v, **end, task=task_id)

        new_graph.add_edge(u, v, **a)

        visited[s_preds] = start
        visited[e_preds] = end
    return new_graph


def merge_and_clean(previous_graph: nx.Graph, new_graph: nx.Graph, task_id: int, classifiers):
    """
    Extract start-end nodes with single edge only.
    :param graph:
    :return:
    """

    if previous_graph is None:
        return extract_pairs(task_id, new_graph)

    predicates = lambda node: tuple(
        sorted([predicate.name for predicate in node['predicates'] if len(predicate.mask) > 0]))
    visited = dict()

    similar = dict()

    for u, v, a in new_graph.edges(data=True):
        start = new_graph.nodes[u]
        end = new_graph.nodes[v]
        s_preds = predicates(start)
        e_preds = predicates(end)
        u = visited.get(s_preds, u)
        v = visited.get(e_preds, v)

        if u not in similar:
            similar[u] = find_similar_nodes(u, previous_graph, classifiers)
        if v not in similar:
            similar[v] = find_similar_nodes(v, previous_graph, classifiers)

        previous_graph = inject(previous_graph, new_graph, u, v, a, similar)
        visited[s_preds] = u
        visited[e_preds] = v
    return previous_graph


def inject(previous_graph, new_graph, start, end, edge, similar):
    start_nodes = similar.get(start, [])
    if len(start_nodes) == 0:
        start_nodes = [start]
        previous_graph.add_node(start, **new_graph.nodes[start])
    end_nodes = similar.get(end, [])
    if len(end_nodes) == 0:
        end_nodes = [end]
        previous_graph.add_node(end, **new_graph.nodes[end])
    for u in start_nodes:
        for v in end_nodes:
            previous_graph.add_edge(u, v, **edge)
    return previous_graph


def get_predicates(graph, node, order=True):
    """
    Extract the predicates from a node in a graph
    """
    predicates = [x for x in graph.nodes[node]['predicates'] if len(x.mask) > 0]
    if not order:
        return predicates
    return sorted(predicates, key=lambda x: x.mask[0])


def find_similar_nodes(node, graph, classifiers):
    """
    Find similar nodes in a new graph
    """

    matches = set()
    misses = set()

    classifiers = classifiers[node]
    if classifiers is None:
        return list()
    current_matches = list()
    for n in graph.nodes:
        predicates = get_predicates(graph, n)
        if len(predicates) != len(classifiers):
            continue
        found = True
        for predicate, classifier in zip(predicates, classifiers):
            if predicate in misses:
                found = False
                break
            elif predicate not in matches:
                data = predicate.sample(100)
                prob = classifier.probability(data, use_mask=False)
                # if prob < 0.05:
                if prob < 0.5:
                    found = False
                    misses.add(predicate)
                    break
                else:
                    matches.add(predicate)
        if found:
            current_matches.append(n)
    return current_matches


def clean_and_fit(classifiers, task, graph):
    """
    Fit classifiers to each node in the graph. The graph nodes will also be relabelled.
    :param classifiers: the existing classifiers
    :param task: the current task
    :param graph: the current graph
    """
    names = {node: "{}:{}".format(task, node) for node in graph.nodes}
    nx.relabel_nodes(graph, names, copy=False)
    predicate_classifier = dict()
    for node in graph.nodes:
        if 'state' in graph.nodes[node]:
            del graph.nodes[node]['state']
        temp = list()
        for predicate in get_predicates(graph, node):
            if predicate in predicate_classifier:
                classifier = predicate_classifier[predicate]
            else:
                data = predicate.sample(1000)
                classifier = OCSupportVectorClassifier(predicate.mask)
                try:
                    classifier.fit(data, use_mask=False)
                except Exception as e:
                    print(data)
                    print(predicate)
                    classifier = None
                    traceback.print_exc()
                predicate_classifier[predicate] = classifier
            temp.append(classifier)
        classifiers[node] = temp
    return classifiers

# def fit_classifiers(classifiers, task, graph):
#     names = {node: "{}:{}".format(task, node) for node in graph.nodes}
#     nx.relabel_nodes(graph, names, copy=False)
#     for node in graph.nodes:
#         for predicate in graph.nodes[node]['predicates']:
#
#             if (task, predicate) in classifiers:
#                 continue
#             data = predicate.sample(1000)
#             classifier = OCSupportVectorClassifier(predicate.mask)
#             try:
#                 classifier.fit(data, use_mask=False)
#             except Exception as e:
#                 print(data)
#                 print(predicate)
#                 classifier = None
#                 traceback.print_exc()
#             classifiers[(task, predicate)] = classifier
#     return classifiers
