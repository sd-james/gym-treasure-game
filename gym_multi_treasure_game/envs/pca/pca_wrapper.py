import random
import time

import gym
import pygame
from gym.envs.classic_control import rendering
import numpy as np

from gym_multi_treasure_game.envs.pca.base_pca import BasePCA, PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from s2s.env.s2s_env import MultiViewEnv, View
from s2s.image import Image


class PCAWrapper(gym.Wrapper):

    def __init__(self, env: MultiViewEnv, pca: BasePCA, pca2: BasePCA = None, expand_dim=False):
        super().__init__(env)
        self._pca = pca
        self._pca2 = pca2
        self._expand_dim = expand_dim

    @property
    def pcas(self):
        return [self._pca, self._pca2]

    def reset(self, **kwargs):
        state, observation = self.env.reset(**kwargs)
        return state, self.observation(observation)

    def step(self, action):
        state, observation, reward, done, info = self.env.step(action)
        return state, self.observation(observation), reward, done, info

    def observation(self, observation):

        if self._pca2 is not None:
            observation = [np.expand_dims(x, axis=0) for x in observation]
            return np.array([self._pca.compress(observation[0]).squeeze(), self._pca2.compress(observation[1]).squeeze()], dtype=object)

        obs = self._pca.compress(observation)
        if not self._expand_dim:
            obs = np.squeeze(obs, axis=0)
        return obs

    def render(self, mode='human', view=View.PROBLEM):
        self.env.drawer.draw_domain()
        local_rgb = None
        if view == View.AGENT:
            # draw the agent view too

            if self._pca2 is None:
                surface = self.drawer.draw_local_view()
                local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame
                local_rgb = self._pca.representation(local_rgb)
            else:
                surface, surface2 = self.drawer.draw_local_view(split=True)
                local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame
                local_rgb2 = pygame.surfarray.array3d(surface2).swapaxes(0, 1)  # swap because pygame
                import matplotlib.pyplot as plt
                plt.imshow(local_rgb)
                plt.show()
                local_rgb = self._pca.representation(local_rgb)
                plt.imshow(local_rgb)
                plt.show()
                local_rgb2 = self._pca2.representation(local_rgb2)
                local_rgb = Image.combine([Image.to_image(local_rgb), Image.to_image(local_rgb2)], mode='L')

        rgb = pygame.surfarray.array3d(self.drawer.screen).swapaxes(0, 1)  # swap because pygame
        if mode == 'rgb_array':
            return local_rgb if view == View.AGENT else rgb
        elif mode == 'human':
            # draw it like gym
            if self.env.viewer is None:
                self.env.viewer = rendering.SimpleImageViewer()

            if view == View.AGENT:
                a = to_image(rgb, mode='RGB')
                b = to_image(local_rgb, mode='L')
                rgb = to_array(combine([a, b]))
            self.env.viewer.imshow(rgb)


if __name__ == '__main__':

    from gym_multi_treasure_game.envs.multi_treasure_game import to_image, to_array, combine, MultiTreasureGame

    random.seed(0)
    np.random.seed(0)
    pca = PCA(PCA_STATE)
    pca.load('models/no_bg_pca_state.dat')

    pca2 = PCA(PCA_INVENTORY)
    pca2.load('models/no_bg_pca_inventory.dat')
    for i in range(1, 11):
        env = PCAWrapper(MultiTreasureGame(i, split_inventory=True, render_bg=False), pca, pca2=pca2)
        state, obs = env.reset()
        env.render('human', view=View.AGENT)
        # print(state)
        for N in range(100):
            mask = env.available_mask
            action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())
            next_state, next_obs, reward, done, info = env.step(action)
            # print(next_state)
            env.render('human', view=View.AGENT)
            if done:
                print("{}: WIN: {}".format(i, N))
                print(info)
                env.close()
                break
            time.sleep(1)
