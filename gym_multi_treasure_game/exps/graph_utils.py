import traceback

import networkx as nx
import numpy as np
from tqdm import tqdm

from s2s.estimators.oc_svc import OCSupportVectorClassifier
from s2s.utils import load

KNN = load('/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/data/knn.pkl')


def extract_options(edge):
    if edge['level'] == 0:
        return [edge['action'].option]

    raise NotImplementedError


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
    start = transition_data['agent_state']
    start2 = transition_data['state']
    end = transition_data['next_agent_state']
    end2 = transition_data['next_state']
    return [(x, y, a, b) for x, y, a, b in zip(start, end, start2, end2)]


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


def similar_predicates(predicate, other_predicate, prev_task, classifiers):
    if predicate.mask != other_predicate.mask:
        return False
    classifier = classifiers[(prev_task, other_predicate)]
    if classifier is None:
        return False
    data = predicate.sample(100)
    prob = classifier.probability(data, use_mask=False)
    return prob > 0.05


def is_match(state, next_state, start_node, end_node):
    def similar(x, y):
        for a, b in zip(x, y):
            if np.linalg.norm(a - b, np.inf) > 1:
                return False
        return True

    # TODO use OC SVM with predicates
    return similar(state, start_node['obs']) and similar(next_state, end_node['obs'])


def _get_all_predicates(graph):
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

def merge(current_graph, previous_graph, transition_data, classifiers):
    if isinstance(previous_graph, list):
        for graph in previous_graph:
            current_graph = merge(current_graph, graph, transition_data, classifiers)
        return current_graph

    # assume current graph already linked
    # use transition data to find nodes to transfer in.
    # link up with problem specific data

    previous_task, previous_graph = previous_graph

    def _exists(node):
        for n in current_graph.nodes:
            if _is_similar(current_graph.nodes[n], node, previous_task, classifiers):
                return True
        return False

    # ignore these ones!
    current_predicates = _get_all_predicates(current_graph)
    prev_predicates = _get_all_predicates(previous_graph)

    duplicate_predicates = dict()
    for previous_predicate in prev_predicates:

        for current in current_predicates:
            if similar_predicates(current, previous_predicate, previous_task, classifiers):
                duplicate_predicates[previous_predicate] = current
                break

    # we now have a mapping from previous to new predicates.
    # Go through node-edge-nodes and integrate into current graph, but replace node with current node if exact match
    old_node_to_new_entry = dict()      # map old node id to id in new graph
    new_nodes = set()
    for u, v, a in previous_graph.edges(data=True):

        match_found, start_node = find_similar_node(previous_graph.nodes[u], duplicate_predicates, current_graph)
        if not match_found:
            # add new node to graph
            if u in old_node_to_new_entry:
                start_node = old_node_to_new_entry[u]
            else:
                new_id = len(current_graph.nodes)
                old_node_to_new_entry[u] = new_id
                start_node = new_id
                new_nodes.add(new_id)
                current_graph.add_node(start_node, **previous_graph.nodes[u])

        match_found, end_node = find_similar_node(previous_graph.nodes[v], duplicate_predicates, current_graph)
        if not match_found:
            # add new node to graph
            if v in old_node_to_new_entry:
                end_node = old_node_to_new_entry[v]
            else:
                new_id = len(current_graph.nodes)
                old_node_to_new_entry[v] = new_id
                end_node = new_id
                new_nodes.add(new_id)
                current_graph.add_node(end_node, **previous_graph.nodes[v])
        current_graph.add_edge(start_node, end_node, **a)

    # use transition data to link
    found_nodes = set()
    for d, dprime, s, sprime in tqdm(_iter(transition_data)):

        # find matching start/end nodes
        for u, v, a in current_graph.edges(data=True):
            if is_match(d, dprime, current_graph.nodes[u], current_graph.nodes[v]):
                found_nodes.add(u)
                found_nodes.add(v)

    for node in current_graph.nodes:
        if node not in found_nodes:
            current_graph.remove_node(node)
    current_graph.remove_nodes_from(list(nx.isolates(current_graph)))



    # exists = {node for node in previous_graph if _exists(previous_graph.nodes[node])}

    # use transition data to find nodes to transfer in.
    matches = list()
    for d, dprime, s, sprime in _iter(transition_data):

        for u, v, a in previous_graph.edges(data=True):
            if u in exists or v in exists:
                # ignore
                continue

            if is_match(d, dprime, previous_graph.nodes[u], previous_graph.nodes[v]):
                matches.append((u, v, a, s, sprime))

    # integrate matches into existing graph!
    for start_node, end_node, edge, s, s_prime in matches:
        pass


def clean(graph):
    for node in graph.nodes:
        del graph.nodes[node]['state']
    return graph


def fit_classifiers(classifiers, task, graph):
    for node in graph.nodes:
        for predicate in graph.nodes[node]['predicates']:

            if (task, predicate) in classifiers:
                continue
            data = predicate.sample(1000)
            classifier = OCSupportVectorClassifier(predicate.mask)
            try:
                classifier.fit(data, use_mask=False)
            except Exception as e:
                print(data)
                print(predicate)
                classifier = None
                traceback.print_exc()
            classifiers[(task, predicate)] = classifier
    return classifiers
