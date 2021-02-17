from collections import defaultdict

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from tqdm import trange

from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from s2s.portable.linking_function import LinkingFunction
from s2s.utils import range_without, make_path, exists, extract_random_episodes, flatten, save, load
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    for task in range_without(4, 11):
        env = MultiTreasureGame(task, split_inventory=True)
        transition_data = pd.DataFrame(
            columns=['state', 'option', 'reward', 'next_state'])
        for ep in trange(50):
            obs = env.reset()[0]
            state = _get_state(env, obs)
            for t in range(1000):
                action = env.sample_action()
                next_obs, _, reward, done, _ = env.step(action)
                next_state = _get_state(env, next_obs)
                transition_data.loc[len(transition_data)] = [state, action, reward, next_state]
                if done:
                    break

                state = next_state
                obs = next_obs

        print("Clustering...")
        knn, means = _cluster(transition_data, epsilon=0.04)
        print("Linking...")
        linking, options = _construct_linking(transition_data, knn)
        graph = _construct_graph(linking, means, options)
        nx.write_gpickle(graph, "graph_{}.pkl".format(task))
