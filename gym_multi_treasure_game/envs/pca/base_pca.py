import pickle

import cv2
import numpy as np

# SIZE = (72, 96)

PCA_STATE = 25
PCA_INVENTORY = 5


class BasePCA:
    def __init__(self, pca):
        self._n_components = self._pca.n_components
        self.from_file = False
        self._components = None
        self._explained_variance_ratio = None
        self._explained_variance = None
        self._mean = None
        self._pca = pca
        self._whiten = False
        self.first_fit = True

    def fit(self, X):
        self._pca.fit(X)

    @property
    def whiten(self):
        return self._whiten

    @property
    def n_components(self):
        if self.from_file:
            return self._n_components
        return self._pca.n_components

    @property
    def explained_variance_ratio(self):
        if self.from_file:
            return self._explained_variance_ratio
        return self._pca.explained_variance_ratio_

    @property
    def explained_variance(self):
        if self.from_file:
            return self._explained_variance
        return self._pca.explained_variance_

    @property
    def components(self):
        if self.from_file:
            return self._components
        return self._pca.components_

    @property
    def mean(self):
        if self.from_file:
            return self._mean
        return self._pca.mean_

    def size(self, image):

        if len(image.shape) == 1:
            # do the inverse
            # TODO: HARD CODED
            N = image.shape[0] // 72
            return 72, N

        return tuple(np.array(image.shape[0:2]) // 2)

    def scale(self, image):

        # resize image
        smaller = cv2.resize(image, self.size(image), interpolation=cv2.INTER_AREA)
        # smaller = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        # plt.imshow(smaller)
        # plt.show()
        return smaller

    def compress_(self, image, preprocess=True):
        if preprocess:
            image = self.scale(image)
            image = self.flat_gray(image)
        X = image - self.mean
        X_transformed = np.dot(X, self.components.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance)
        return X_transformed

    def representation(self, image, **kwargs):

        # get a PCA representation of the image
        size = self.size(image)
        y = self.compress_(image, **kwargs)
        decoded = self.uncompress_(y).reshape(size[::-1])
        return np.uint8(decoded)

    def uncompress_(self, image):

        if self.whiten:
            return np.dot(image, np.sqrt(self.explained_variance[:, np.newaxis]) * self.components) + self.mean
        else:
            return np.dot(image, self.components) + self.mean

    def shrink(self, state):

        n_non_images = 0
        compress = list()
        for i in range(state.shape[0] - n_non_images):
            smaller = cv2.resize(state[i], self.size(state[i]), interpolation=cv2.INTER_AREA)
            compress.append(smaller)

        for i in range(-n_non_images, 0, 1):
            compress.append(state[i])
        compressed = np.array(compress)
        return compressed

    def compress(self, state, flatten=False):
        compress = list()
        for i in range(state.shape[0]):
            if len(state[i].shape) == 3:
                compress.append(self.compress_(state[i]))
            else:
                compress.append(state[i])
        compressed = np.array(compress)
        if flatten:
            return np.concatenate(compressed).ravel()
        return compressed

    def uncompress(self, compressed_state):
        uncompress = list()
        for i in range(compressed_state.shape[0]):
            uncompress.append(self.unflatten(self.uncompress_(compressed_state[i])))
        # for i in range(-n_non_images, 0, 1):
        #     uncompress.append(compressed_state[i])
        return np.array(uncompress)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.n_components, self.explained_variance_ratio, self.components, self.mean, self.first_fit,
                         self._pca.explained_variance_, self.whiten),
                        file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self._n_components, self._explained_variance_ratio, self._components, self._mean, self.first_fit, \
            self._explained_variance, self._whiten = pickle.load(file)

            self.from_file = True

    def rgb2gray_(self, rgb):
        return np.uint8(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]))

    def unflatten(self, image):
        return np.reshape(image, self.size(image)[::-1])

    def flat_gray(self, image):
        return np.reshape(self.rgb2gray_(image), (image.shape[0] * image.shape[1]))

    def add_state_(self, X, idx, state):

        state = np.expand_dims(state, axis=0)

        n_non_ims = 0
        for i in range(state.shape[0] - n_non_ims):  # -2 because last one xy and second last is inventory

            image = cv2.resize(state[i], self.size(state[i]), interpolation=cv2.INTER_AREA)

            X[idx, :] = np.reshape(self.rgb2gray_(image), (image.shape[0] * image.shape[1]))
            idx += 1
        return idx

    def extract_(self, trajectory):
        first = True
        j = 0

        # n_samples = 100
        # trajectory = random.sample(trajectory, n_samples) + [trajectory[-1]]

        for state, action, reward, done, next_state in trajectory:
            if first:
                first = False
                # -2 because the last is the position and second last is inventory
                size = self.size(state)
                n_non_ims = 0
                X = np.zeros(
                    shape=((len(trajectory) + 1)
                           # * (state.shape[0] - n_non_ims)
                           , size[0] * size[1]))
                j = self.add_state_(X, j, state)
            j = self.add_state_(X, j, next_state)
        return X

    def fit_transitions(self, episodes):
        print("HERE")
        data = list()
        for episode in episodes:
            data.append(self.extract_(episode))
        print("Extracted data.bak")
        X = np.vstack(tuple(data))
        print(X.shape)
        print("Fitting data.bak")
        self.fit(X)

        print("Data fitted")
        print(self.explained_variance_ratio)
