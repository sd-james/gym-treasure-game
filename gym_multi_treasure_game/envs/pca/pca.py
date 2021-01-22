from sklearn.decomposition import PCA as PCA_

from gym_multi_treasure_game.envs.pca.base_pca import BasePCA


class PCA(BasePCA):
    """
    PCA implementation
    """

    def __init__(self, n_components, whiten=True):
        """
        Create an Incremental PCA implementation
        :param n_components:
        :param batch_size:
        :param whiten:
        """
        self._pca = PCA_(n_components=n_components, copy=False, whiten=whiten)
        super().__init__(self._pca)
        self._whiten = whiten
