import random
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from tqdm import trange

from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from s2s.portable.linking_function import LinkingFunction
from s2s.utils import range_without, flatten, save


def _get_state(env, obs):
    x = obs[0]
    y = obs[1]
    if len(obs) == 3:
        door_closed = int(obs[2] > 0.5)
    else:
        door_closed = 0
    has_key = int(env._env.player_got_key())
    has_gold = int(env._env.player_got_goldcoin())
    return np.array([x, y, door_closed, has_key, has_gold])


def _cluster(data: pd.DataFrame, epsilon):
    states = flatten(data['state'])
    next_states = flatten(data['next_state'])
    data = np.vstack((states, next_states))
    db = DBSCAN(eps=epsilon, min_samples=3)
    labels = db.fit_predict(data)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(data, labels)

    means = dict()
    for label in set(labels):
        if label == -1:
            continue
        points_of_cluster = data[labels == label, :]
        centroid_of_cluster = np.mean(points_of_cluster, axis=0)
        means[label] = centroid_of_cluster

    return neigh, means


def _construct_linking(data, knn):
    link = LinkingFunction()
    options = defaultdict(list)
    for _, row in data.iterrows():
        state = row['state'].reshape(1, -1)
        next_state = row['next_state'].reshape(1, -1)
        start = knn.predict(state)[0]
        end = knn.predict(next_state)[0]
        if start == -1 or end == -1:
            continue
        link.add(start, end)
        options[(start, end)].append(row['option'])
    return link, options


def _construct_graph(link, means, options):
    graph = nx.DiGraph()

    for start, end, prob in link:
        graph.add_edge(start, end, weight=prob, options=options[(start, end)])

    positions = {}
    for label, mean in means.items():

        pos = np.array([mean[0], 1 - mean[1]])
        if mean[2] == 1:
            pos += [0.01, -0.02]
        if mean[3] == 1:
            pos += [0.03, 0.03]
        if mean[4] == 1:
            pos += [-0.02, -0.02]

        graph.nodes[label]['state'] = np.array(mean)
        graph.nodes[label]['pos'] = pos
        positions[label] = tuple(pos)
    # nx.draw(graph, pos=positions, node_size=30)
    # plt.show()
    return graph


def _get_random_path(graph, used):
    while True:
        start = random.choice(list(graph.nodes))
        end = random.choice(list(graph.nodes))
        if start == end or (start, end) in used:
            continue
        if nx.has_path(graph, start, end):
            return nx.shortest_path(graph, start, end)


def _extract_plan(graph, path):
    path_graph = nx.path_graph(path)  # does not pass edges attributes

    # Read attributes from each edge
    plan = list()
    for ea in path_graph.edges():

        options = graph.edges[ea[0], ea[1]]['options']
        if len(set(options)) > 1:
            raise ValueError
        plan.append(options[0])
    return plan


if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)
    test_cases = defaultdict(list)
    for task in range_without(1, 11):
        used = set()
        graph = nx.read_gpickle("../data/ground_truth/graph_{}.pkl".format(task))
        for _ in range(100):
            path = _get_random_path(graph, used)
            used.add((path[0], path[-1]))
            test_cases[task].append((path[0], _extract_plan(graph, path), path[-1]))

    save(test_cases, '../data/test_cases.pkl')
