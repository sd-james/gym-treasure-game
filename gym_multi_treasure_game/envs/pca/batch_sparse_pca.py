from sklearn.decomposition import MiniBatchSparsePCA

from gym_multi_treasure_game.envs.pca import BasePCA


class BatchSparsePCA(BasePCA):
    def __init__(self, n_components, batch_size):
        self._pca = MiniBatchSparsePCA(n_components=n_components, batch_size=batch_size, normalize_components=True,
                                       verbose=2)
        super().__init__(self._pca)