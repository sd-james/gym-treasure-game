import random
import time

import gym
import pygame
from gym.envs.classic_control import rendering
import numpy as np

from gym_multi_treasure_game.envs.pca.base_pca import BasePCA, PCA_N
from gym_multi_treasure_game.envs.pca.pca import PCA
from s2s.env.s2s_env import MultiViewEnv, View


class PCAWrapper(gym.Wrapper):

    def __init__(self, env: MultiViewEnv, pca: BasePCA, expand_dim=False):
        super().__init__(env)
        self._pca = pca
        self._expand_dim = expand_dim

    def reset(self, **kwargs):
        state, observation = self.env.reset(**kwargs)
        return state, self.observation(observation)

    def step(self, action):
        state, observation, reward, done, info = self.env.step(action)
        return state, self.observation(observation), reward, done, info

    def observation(self, observation):
        obs = self._pca.compress(observation)
        if not self._expand_dim:
            obs = np.squeeze(obs, axis=0)
        return obs

    def render(self, mode='human', view=View.PROBLEM):
        self.env.drawer.draw_domain()
        local_rgb = None
        if view == View.AGENT:
            # draw the agent view too
            surface = self.drawer.draw_local_view()
            local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame
            local_rgb = self._pca.representation(local_rgb)

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
    pca = PCA(PCA_N)
    pca.load('models/20_runs_pca_30_empty_inv.dat')
    for i in range(1,  11):
        env = PCAWrapper(MultiTreasureGame(i), pca)
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
