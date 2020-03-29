import os
from typing import Any, List

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # sorry PyGame

import gym
import numpy as np

import pygame
from gym.envs.classic_control import rendering
from gym.spaces import Discrete, Box

from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_drawer import _TreasureGameDrawer
from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_impl import _TreasureGameImpl, create_options

__author__ = 'Steve James and George Konidaris'


def make_env(id: str, **kwargs):
    """
    Convenience function for calling gym.make
    """
    return gym.make(id, **kwargs)


def make_path(root,
              *args):
    """
    Creates a path from the given parameters
    :param root: the root of the path
    :param args: the elements of the path
    :return: a string, each element separated by a forward slash.
    """
    path = root
    if path.endswith('/'):
        path = path[0:-1]
    for element in args:
        if not isinstance(element, str):
            element = str(element)
        if element[0] != '/':
            path += '/'
        path += element
    return path

def get_dir_name(file):
    """
    Get the directory of the given file
    :param file: the file
    :return: the file's directory
    """
    return os.path.dirname(os.path.realpath(file))


class ObservationWrapper(gym.Wrapper):

    def reset(self, **kwargs):
        super().reset(**kwargs)
        # use the render function to get the screen
        screen = self.render(mode='rgb_array')
        return screen

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['world_state'] = obs
        # use the render function to get the screen
        screen = self.render(mode='rgb_array')
        return screen, reward, done, info


class TreasureGame(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    """
    The Treasure Game as a Gym environment. For more details of the environment, see
    G.D. Konidaris, L.P. Kaelbling, and T. Lozano-Perez. From Skills to Symbols: Learning Symbolic Representations for
    Abstract High-Level Planning. Journal of Artificial Intelligence Research 61, pages 215-289, January 2018
    """

    def __init__(self):
        """
        Create a new instantiation of the Treasure Game
        """
        dir = os.path.dirname(os.path.realpath(__file__))
        dir = make_path(dir, '_treasure_game_impl')
        self._env = _TreasureGameImpl(make_path(dir, 'domain.txt'), make_path(dir, 'domain-objects.txt'),
                                      make_path(dir, 'domain-interactions.txt'))
        self.drawer = None
        self.option_list, self.option_names = create_options(self._env)
        self.action_space = Discrete(len(self.option_list))
        s = self._env.get_state()
        self.observation_space = Box(np.float32(0.0), np.float32(1.0), shape=(len(s),))
        self.viewer = None
        self.local_viewer = None

    def reset(self):
        self._env.reset_game()
        self.option_list, self.option_names = create_options(self._env, None)
        return self._env.get_state()

    @property
    def available_mask(self):
        """
        Get a binary-encoded array of the options that can be run at the current state
        :return: a binary array specifying which options can be run
        """
        return np.array([int(o.can_run()) for o in self.option_list])

    def step(self, action):
        option = self.option_list[action]
        r = option.run()
        state = self._env.get_state()
        done = self._env.player_got_goldcoin() and self._env.get_player_cell()[1] == 0  # got gold and returned
        return state, r, done, {}

    def render(self, mode='human'):
        if self.drawer is None:
            self.drawer = _TreasureGameDrawer(self._env)

        self.drawer.draw_domain()
        rgb = pygame.surfarray.array3d(self.drawer.screen).swapaxes(0, 1)  # swap because pygame
        if mode == 'rgb_array':
            return rgb
        elif mode == 'human':
            # draw it like gym
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(rgb)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


if __name__ == '__main__':

    env = TreasureGame()
    for episode in range(5):
        state = env.reset()
        for _ in range(1000):
            mask = env.available_mask
            action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())
            next_state, reward, done, _ = env.step(action)
            env.render('human')
            if done:
                break
