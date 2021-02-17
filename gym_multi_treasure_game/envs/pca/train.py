import multiprocessing
import random
import warnings
from functools import partial

import numpy as np
from tqdm import trange
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.sparse_pca import SparsePCA
from s2s.utils import save, load, run_parallel, range_without


def collect(task):
    episodes = list()
    episodes2 = list()
    for _ in trange(25):
        env = MultiTreasureGame(task, split_inventory=True, fancy_graphics=False, render_bg=True)
        solved = False
        while not solved:
            episode = list()
            episode2 = list()
            state, obs = env.reset()
            # print(state)
            for N in range(1000):
                mask = env.available_mask
                action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())

                next_state, next_obs, reward, done, info = env.step(action)
                # import matplotlib.pyplot as plt
                # plt.imshow(next_obs[0])
                # plt.show()
                episode.append((obs[0], action, reward, done, next_obs[0]))
                episode2.append((obs[1], action, reward, done, next_obs[1]))

                # if action == 4:
                #     import matplotlib.pyplot as plt
                #     temp = env.current_agent_observation()
                #     plt.imshow(obs[1])
                #     plt.show()
                #     plt.imshow(next_obs[1])
                #     plt.show()

                obs = next_obs
                state = next_state
                # env.render('human', view=View.AGENT)
                if done:
                    # print("{}: WIN: {}".format(i, N))
                    # print(info)
                    solved = True
                    env.close()
                    break
            episodes.append(episode)
            episodes2.append(episode2)
    print('{} DONE'.format(task))
    return episodes, episodes2


if __name__ == '__main__':

    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    warnings.filterwarnings("ignore")

    functions = [partial(collect, i) for i in range_without(1, 11)]
    results = run_parallel(functions, serial=False)
    episodes = sum([x for x, _ in results], [])
    episodes2 = sum([y for _, y in results], [])

    print('SAVING...')
    #
    save((episodes, episodes2), 'aa.pkl')
    episodes, episodes2 = load('aa.pkl')

    print('FITTING...')
    #
    # temp = load()
    # for x in reversed(temp):
    #     plt.imshow(x)
    #     plt.show()

    # temp = list()
    # for x in episodes2:
    #     for s, a, r, d, s_prime in x:
    #         temp.append(s_prime)
    # temp = np.array(temp)[np.unique(temp)]
    # save(temp)
    # print(len(temp))
    # exit(0)

    # pca = PCA(PCA_STATE)
    # pca.fit_transitions(episodes)
    # pca.save('models/dropped_key_pca_state.dat')

    pca = PCA(PCA_INVENTORY)
    pca.fit_transitions(episodes2)
    pca.save('models/dropped_key_pca_inventory.dat')
