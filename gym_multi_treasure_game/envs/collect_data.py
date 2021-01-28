import numpy as np
from tqdm import trange

from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from s2s.core.build_model import build_model
from s2s.core.explore import collect_data
from s2s.env.s2s_env import S2SWrapper
from s2s.utils import make_dir

if __name__ == '__main__':

    for task in trange(7, 11):
        pca = PCA(PCA_STATE)

        pca.load('pca/models/mod_no_fancy_pca_state.dat')

        pca2 = PCA(PCA_INVENTORY)

        pca2.load('pca/models/mod_no_fancy_pca_inventory.dat')

        env = PCAWrapper(MultiTreasureGame(task, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
        save_dir = '../data.bak/{}'.format(task)
        make_dir(save_dir, clean=False)

        transition_data, initiation_data = collect_data(S2SWrapper(env, options_per_episode=1000),
                                                        n_jobs=16,
                                                        seed=0,
                                                        verbose=True,
                                                        max_episode=30)
        transition_data.to_pickle('{}/transition.pkl'.format(save_dir), compression='gzip')
        initiation_data.to_pickle('{}/init.pkl'.format(save_dir), compression='gzip')
