import random

import gym
import numpy as np
import pygame
from PIL import Image as I
from gym.envs.classic_control import rendering
from gym.spaces import Discrete, Box, MultiDiscrete

from gym_multi_treasure_game.envs import TreasureGame
from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_drawer import _TreasureGameDrawer
from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_impl import _TreasureGameImpl, create_options
from gym_multi_treasure_game.envs.multiview_env import MultiViewEnv, View
from gym_multi_treasure_game.envs.treasure_game import make_path, get_dir_name


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


class MultiTreasureGame(MultiViewEnv, TreasureGame):

    def __init__(self, version_number: int):

        if version_number == 0:
            # use original
            super().__init__()
            return

        self._version_number = version_number
        dir = make_path(get_dir_name(__file__), 'layouts')
        self._env = _TreasureGameImpl(make_path(dir, 'domain_v{}.txt'.format(version_number)),
                                      make_path(dir, 'domain-objects_v{}.txt'.format(version_number)),
                                      make_path(dir, 'domain-interactions_v{}.txt'.format(version_number)))
        self.drawer = None
        self.option_list, self.option_names = create_options(self._env)
        self.action_space = Discrete(len(self.option_list))
        s = self._env.get_state()
        self.observation_space = Box(np.float32(0.0), np.float32(1.0), shape=(len(s),))
        self.viewer = None

    @property
    def agent_space(self) -> MultiDiscrete:
        return MultiDiscrete([11] * 9 + [2, 2])  # 9 for the cells, then boolean on has key, has coin

    def n_dims(self, view: View) -> int:
        """
        The dimensionality of the state space, depending on the view
        """
        if view == View.PROBLEM:
            return self.observation_space.shape[-1]
        return len(self.agent_space.nvec)

    def current_agent_observation(self) -> np.ndarray:
        return self._env.current_observation()

    def __str__(self):
        return "TreasureGameV{}".format(self._version_number)

    def describe_option(self, option: int) -> str:
        return self.option_names[option]

    def render(self, mode='human', view=View.PROBLEM):
        if self.drawer is None:
            self.drawer = _TreasureGameDrawer(self._env)

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


if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    for i in range(1, 11):
        env = MultiTreasureGame(i)
        solved = False
        while not solved:
            state, obs = env.reset()
            for N in range(1000):
                mask = env.available_mask
                action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())
                next_state, next_obs, reward, done, info = env.step(action)
                env.render('human', view=View.AGENT)
                if done:
                    print("WIN: {}".format(N))
                    print(info)
                    solved = True
                    env.close()
                    break
                # time.sleep(0.5)