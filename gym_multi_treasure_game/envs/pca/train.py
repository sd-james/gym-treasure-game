import multiprocessing
import warnings
from functools import partial

import numpy as np
from tqdm import trange
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_N
from gym_multi_treasure_game.envs.pca.pca import PCA


def run_parallel(functions, serial=False):
    """
    Run the list of function in parallel and return the results in a list
    :param functions: the functions to execute
    :return: a list of results
    """
    n_procs = len(functions)

    if serial:
        return [functions[i]() for i in range(n_procs)]

    pool = multiprocessing.Pool(processes=n_procs)
    processes = [pool.apply_async(functions[i]) for i in range(n_procs)]
    return [p.get() for p in processes]


def collect(task):
    episodes = list()
    for _ in trange(20):
        env = MultiTreasureGame(task)
        solved = False
        while not solved:
            episode = list()
            state, obs = env.reset()
            # print(state)
            for N in range(1000):
                mask = env.available_mask
                action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())

                next_state, next_obs, reward, done, info = env.step(action)
                episode.append((obs, action, reward, done, next_obs))
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
    print('{} DONE'.format(task))
    return episodes


if __name__ == '__main__':

    # X = np.load('data.npy')
    # pca = PCA(PCA_N)
    # pca.load('pca.dat')
    #
    # for _ in range(10):
    #     i = np.random.randint(0, X.shape[0])
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1, 2, 1)
    #     ax1.imshow(pca.representation(X[i, :], preprocess=False))
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     x = np.reshape(X[i, :], SIZE).astype(np.uint8)
    #     x = np.swapaxes(x, 0, 1)
    #     ax2.imshow(pca.representation(X[i, :], preprocess=False))
    #     plt.show()

    warnings.filterwarnings("ignore")

    functions = [partial(collect, i) for i in range(1, 11)]
    episodes = sum(run_parallel(functions), [])

    # for i in trange(10, 11):
    #     for _ in trange(10):
    #         env = MultiTreasureGame(i)
    #         solved = False
    #
    #         while not solved:
    #             episode = list()
    #             state, obs = env.reset()
    #             # print(state)
    #             for N in range(1000):
    #                 mask = env.available_mask
    #                 action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())
    #
    #                 next_state, next_obs, reward, done, info = env.step(action)
    #                 episode.append((obs, action, reward, done, next_obs))
    #                 obs = next_obs
    #                 state = next_state
    #                 env.render('human', view=View.AGENT)
    #                 if done:
    #                     print("{}: WIN: {}".format(i, N))
    #                     print(info)
    #                     solved = True
    #                     env.close()
    #                     break
    #             episodes.append(episode)

    print('FITTING...')

    pca = PCA(PCA_N)
    pca.fit_transitions(episodes)

    pca.save('20_runs_pca_30_empty_inv.dat')

    # with open('temp.dat', 'wb') as file:
    #     pickle.dump(episodes, file)
