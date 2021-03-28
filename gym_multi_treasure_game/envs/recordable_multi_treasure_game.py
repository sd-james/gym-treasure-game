import cv2
import pygame
from gym.envs.classic_control import rendering
import numpy as np
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame, to_image, to_array, combine
from gym_multi_treasure_game.envs.treasure_game_impl_.treasure_game_drawer import TreasureGameDrawer_
from gym_multi_treasure_game.envs.treasure_game_impl_.treasure_game_impl import create_options, TreasureGameImpl_
from s2s.env.s2s_env import View
from s2s.image import Image


class RecordableDrawer(TreasureGameDrawer_):

    def __init__(self, recorder, env, global_only=False, alpha=False, redraw=True):
        super().__init__(env, fancy_graphics=True)
        self.recorder = recorder
        self._global_only = global_only
        self._alpha = alpha
        self._redraw = redraw

    def draw_domain(self, show_screen=True, **kwargs):
        super().draw_domain(show_screen=show_screen, alpha=self._alpha, redraw=kwargs.get('redraw', self._redraw))

        if self.recorder.pcas is not None:
            surface, surface2 = self.draw_local_view(split=True)
            local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame
            local_rgb2 = pygame.surfarray.array3d(surface2).swapaxes(0, 1)  # swap because pygame
            local_rgb = self.recorder.pcas[0].representation(local_rgb)
            local_rgb2 = self.recorder.pcas[1].representation(local_rgb2)
            local_rgb = Image.combine([Image.to_image(local_rgb), Image.to_image(local_rgb2)], mode='L')
            local_rgb = np.stack((local_rgb,) * 3, axis=-1)
        else:
            surface = self.draw_local_view()
            local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame

        rgb = pygame.surfarray.array3d(self.screen).swapaxes(0, 1)  # swap because pygame
        a = to_image(rgb, mode='RGB')
        if self._global_only:
            rgb = a
        else:
            b = to_image(local_rgb, mode='RGB')
            rgb = to_array(combine([a, b]))
        self.recorder.views.append(rgb)


class RecordableMultiTreasureGame(MultiTreasureGame):

    def __init__(self, version_number: int, pcas=None, global_only=False, alpha=False, redraw=True):
        super().__init__(version_number, True, pcas, fancy_graphics=True)
        self.drawer = RecordableDrawer(self, self._env, global_only=global_only, alpha=alpha, redraw=redraw)
        # self.drawer = TreasureGameDrawer_(self._env, fancy_graphics=True)
        self.pcas = pcas
        self.option_list, self.option_names = create_options(self._env, self.drawer)
        self.views = list()

    def render(self, mode='human'):
        if self.drawer is None:
            raise ValueError

        self.drawer.draw_domain()
        # draw the agent view too

        rgb = pygame.surfarray.array3d(self.drawer.screen).swapaxes(0, 1)  # swap because pygame

        if self.pcas is not None:
            surface, surface2 = self.drawer.draw_local_view(split=True)
            local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame
            local_rgb2 = pygame.surfarray.array3d(surface2).swapaxes(0, 1)  # swap because pygame
            local_rgb = self.pcas[0].representation(local_rgb)
            local_rgb2 = self.pcas[1].representation(local_rgb2)
            local_rgb = Image.combine([Image.to_image(local_rgb), Image.to_image(local_rgb2)], mode='L')
            local_rgb = np.stack((local_rgb,) * 3, axis=-1)



        else:
            surface = self.drawer.draw_local_view()
            local_rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)  # swap because pygame

        if mode == 'rgb_array':
            return local_rgb
        elif mode == 'human':
            # draw it like gym
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            a = to_image(rgb, mode='RGB')
            b = to_image(local_rgb, mode='RGB')
            rgb = to_array(combine([a, b]))
            self.viewer.imshow(rgb)
            return rgb

    def reset_view(self):
        self.drawer.screen.fill((0, 0, 0))
        self.drawer.draw_domain(show_screen=True, redraw=True)
        self.views = list()

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        self.reset_view()
        self.option_list, self.option_names = create_options(self._env, self.drawer)

        self.render()
        return ret

    def step(self, action):
        ret = super().step(action)
        self.render()
        return ret
