from tqdm import trange

from tqdm import trange

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from s2s.core.build_model import build_model
from s2s.core.explore import collect_data
from s2s.env.s2s_env import S2SWrapper, View
from s2s.utils import make_dir, exists

if __name__ == '__main__':

    for i in trange(5):
        for n_episodes in trange(1,2):
            for task in trange(1, 11):
                    pca = PCA(PCA_STATE)
                    pca.load('pca/models/dropped_key_pca_state.dat')
                    pca2 = PCA(PCA_INVENTORY)
                    pca2.load('pca/models/dropped_key_pca_inventory.dat')
                    env = PCAWrapper(MultiTreasureGame(task, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
                    save_dir = '/media/hdd/treasure_data/{}/{}/{}'.format(task, i, n_episodes)
                    make_dir(save_dir, clean=False)
                    domain, problem, info = build_model(env,
                                                        early_stop=True,
                                                        reload=False,
                                                        save_dir=save_dir,
                                                        load_existing_data='/media/hdd/treasure_data/{}/{}'.format(task, i),
                                                        n_jobs=16,
                                                        # seed=0,
                                                        n_episodes=n_episodes,
                                                        options_per_episode=1000,
                                                        view=View.AGENT,
                                                        **CONFIG[task],
                                                        visualise=True,
                                                        verbose=True)

