from sklearn.decomposition import SparsePCA as SPCA

import pickle
import numpy as np

from gym_multi_treasure_game.envs.pca.base_pca import BasePCA


class SparsePCA(BasePCA):

    def __init__(self, n_components, normalise_images=False, alpha=1):
        self._pca = SPCA(n_components=n_components, alpha=alpha, verbose=1, n_jobs=8)
        super().__init__(self._pca)
        self.normalise_images = normalise_images

    def fit(self, X):
        if self.normalise_images:
            X = X.astype(np.float32) / 255.0
        self._pca.fit(X)

    def compress_(self, image, preprocess=True):

        if preprocess:
            image = self.scale(image)
            image = self.flat_gray(image)
        if self.normalise_images:
            image = image.astype(np.float32) / 255.0
        X = image - self.mean
        X_transformed = np.dot(X, self.components.T)
        return X_transformed

    def uncompress_(self, image):

        uncompressed = super().uncompress_(image)
        if self.normalise_images:
            uncompressed = np.uint8(uncompressed * 255)
        return uncompressed

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.n_components, self.components, self.mean, self.normalise_images),
                        file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self._n_components, self._components, self._mean, self.normalise_images = pickle.load(file)
            self.from_file = True
