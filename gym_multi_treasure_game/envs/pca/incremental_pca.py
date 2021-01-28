from sklearn.decomposition import IncrementalPCA as IPCA

from gym_multi_treasure_game.envs.pca.base_pca import BasePCA


class IncrementalPCA(BasePCA):
    """
    Incremental PCA implementation
    """

    def __init__(self, n_components, batch_size=30, whiten=False):
        """
        Create an Incremental PCA implementation
        :param n_components:
        :param batch_size:
        :param whiten:
        """
        self._pca = IPCA(n_components=n_components, batch_size=batch_size, copy=False, whiten=whiten)
        super().__init__(self._pca)

        self.first_fit = True
        self._whiten = whiten



    def fit(self, X):
        if self.first_fit:
            self.first_fit = False
            self._pca.fit(X)
        else:
            self._pca.partial_fit(X)
