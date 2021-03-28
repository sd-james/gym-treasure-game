from tqdm import trange

from tqdm import trange

from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from s2s.core.explore import collect_data
from s2s.env.s2s_env import S2SWrapper
from s2s.utils import make_dir, exists

if __name__ == '__main__':

    for i in trange(5, 10):
        for n_episodes in trange(50, 51):
            for task in trange(1,11):
                try:
                    pca = PCA(PCA_STATE)

                    pca.load('pca/models/dropped_key_pca_state.dat')

                    pca2 = PCA(PCA_INVENTORY)

                    pca2.load('pca/models/dropped_key_pca_inventory.dat')

                    env = PCAWrapper(MultiTreasureGame(task, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
                    save_dir = '/media/hdd/treasure_data/{}/{}/{}'.format(task, i, n_episodes)
                    make_dir(save_dir, clean=False)

                    if exists('{}/transition.pkl'.format(save_dir)) and exists('{}/init.pkl'.format(save_dir)):
                        pass
                    else:
                        print("In {}".format(save_dir))
                        transition_data, initiation_data = collect_data(S2SWrapper(env, options_per_episode=1000),
                                                                        n_jobs=17,
                                                                        seed=None,
                                                                        verbose=True,
                                                                        max_episode=n_episodes)
                        transition_data.to_pickle('{}/transition.pkl'.format(save_dir), compression='gzip')
                        initiation_data.to_pickle('{}/init.pkl'.format(save_dir), compression='gzip')
                except:
                    pass
