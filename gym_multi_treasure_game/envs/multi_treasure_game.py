from typing import List

from gym.spaces import Discrete
from tqdm import trange

from gym_multi_treasure_game.envs import TreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import BasePCA, PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.treasure_game import make_path, get_dir_name
from gym_multi_treasure_game.envs.treasure_game_impl_.treasure_game_drawer import TreasureGameDrawer_
from gym_multi_treasure_game.envs.treasure_game_impl_.treasure_game_impl import TreasureGameImpl_, create_options
from s2s.env.s2s_env import MultiViewEnv, View, S2SEnv
import matplotlib.pyplot as plt

from s2s.image import Image


def to_image(image, mode='L'):
    img = I.fromarray(np.uint8(image), mode=mode)
    return img


def to_array(image):
    return np.array(image.convert('RGB'))


def combine(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    sum_height = sum(heights)
    new_im = I.new('RGB', (max_width, sum_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


import random
import time

import numpy as np
import pygame
from PIL import Image as I
from gym.envs.classic_control import rendering
from gym.spaces import Box


def to_image(image, mode='L'):
    img = I.fromarray(np.uint8(image), mode=mode)
    return img


def to_array(image):
    return np.array(image.convert('RGB'))


def combine(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    sum_height = sum(heights)
    new_im = I.new('RGB', (max_width, sum_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


class MultiTreasureGame(MultiViewEnv, TreasureGame, S2SEnv):

    @property
    def available_mask(self) -> np.ndarray:
        return super().available_mask

    def __init__(self, version_number: int, split_inventory=False, pcas: List[BasePCA] = None, fancy_graphics=False,
                 render_bg=True):
        self._version_number = version_number

        if version_number == 0:
            # use original
            print("Disabling original game. Use versions > 0")
            exit(-1)
            super().__init__()
            return

        self._split_inventory = split_inventory
        self._pcas = pcas
        dir = make_path(get_dir_name(__file__), 'layouts')
        self._env = TreasureGameImpl_(make_path(dir, 'domain_v{}.txt'.format(version_number)),
                                      make_path(dir, 'domain-objects_v{}.txt'.format(version_number)),
                                      make_path(dir, 'domain-interactions_v{}.txt'.format(version_number)))
        self.drawer = None
        self.option_list, self.option_names = create_options(self._env)
        self.action_space = Discrete(len(self.option_list))
        s = self._env.get_state()
        self.observation_space = Box(np.float32(0.0), np.float32(1.0), shape=(len(s),))
        self.viewer = None
        self.fancy_graphics = fancy_graphics
        self.render_background = render_bg

    @property
    def version(self):
        return self._version_number

    def __str__(self):
        return "TreasureGameV{}".format(self._version_number)

    def _render_state(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an image of the given state. There should be no missing state variables (using render_state if so)
        """

        if kwargs.get('view', View.PROBLEM) == View.PROBLEM:

            if len(state) > 2 and state[-1] > 0.5:
                colour = pygame.Color('red')
            else:
                colour = pygame.Color('green')
            surface = self.drawer.draw_background_to_surface()
            surface.set_alpha(32)

            left = state[0] * surface.get_width()
            top = state[1] * surface.get_height()
            w = 32
            h = 32

            pygame.draw.rect(surface, colour, (left, top, w, h))
            image = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        else:

            mask = kwargs.get('mask', [0, 1])
            pcas = np.array(self._pcas)[mask]
            if not kwargs.get('masked', False):
                state = state[mask]

            x = [pca.unflatten(pca.uncompress_(s)) for pca, s in zip(pcas, state)]
            x = map(Image.to_image, x)
            image = Image.combine(list(x), mode='L')
            image = Image.to_array(image, mode='L')
        return image

    @property
    def agent_space(self) -> Box:
        return Box(0, 255, (192, 144, 3))

    def describe_option(self, option: int) -> str:
        return self.option_names[option]

    def n_dims(self, view: View, flat=False) -> int:
        """
        The dimensionality of the state space, depending on the view
        """
        if view == View.PROBLEM:
            return self.observation_space.shape[-1]
        if flat:
            return PCA_STATE + PCA_INVENTORY
        return 2

    @property
    def underlying_state(self):
        state_vec = []
        (cell_x, cell_y) = self._env.get_player_cell()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                state_vec.append(self._env._get_object(cell_x + dx, cell_y + dy))
        return np.array([np.array(state_vec),
                         np.array([int(self._env.player_got_key()), int(self._env.player_got_goldcoin())])
                         ], dtype=object)

    def current_agent_observation(self) -> np.ndarray:
        if self._split_inventory:
            return np.array(self._env.current_observation(self.drawer, split=self._split_inventory), dtype=object)
        return np.expand_dims(self._env.current_observation(self.drawer),
                              axis=0)  # 1 x (width x height x channels)

    def render(self, mode='human', view=View.PROBLEM):
        if self.drawer is None:
            self.drawer = TreasureGameDrawer_(self._env)

        self.drawer.draw_domain()
        local_rgb = None
        if view == View.AGENT:
            # draw the agent view too
            surface = self.drawer.draw_local_view()
            local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame

        rgb = pygame.surfarray.array3d(self.drawer.screen).swapaxes(0, 1)  # swap because pygame
        if mode == 'rgb_array':
            return local_rgb if view == View.AGENT else rgb
        elif mode == 'human':
            # draw it like gym
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            if view == View.AGENT:
                a = to_image(rgb, mode='RGB')
                b = to_image(local_rgb, mode='RGB')
                rgb = to_array(combine([a, b]))
            self.viewer.imshow(rgb)
            return rgb

    @staticmethod
    def animate(version, pcas, plan):
        def _get_options(operator):
            if 'options' in operator.data:
                return operator.data['options']
            return [operator.option]

        from gym_multi_treasure_game.envs.recordable_multi_treasure_game import RecordableMultiTreasureGame
        env = RecordableMultiTreasureGame(version, pcas=pcas)
        env.reset()
        for operator in plan:
            for option in _get_options(operator):
                env.step(option)
        env.render()
        env.close()
        return env.views

    def close(self):
        pygame.quit()
        super().close()

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    for i in range(3, 10):
        env = MultiTreasureGame(i, split_inventory=True, render_bg=True)
        solved = False
        for ep in trange(50):
            state, obs = env.reset()
            # print(state)
            for N in range(1000):
                mask = env.available_mask
                action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())
                next_state, next_obs, reward, done, info = env.step(action)
                # print(next_state)

                env.render('human', view=View.AGENT)
                if done:
                    print("{}: WIN: {}".format(i, N))
                    print(info)
                    # time.sleep(10)
                    solved = True
                    # env.close()
                    break
                time.sleep(1)
