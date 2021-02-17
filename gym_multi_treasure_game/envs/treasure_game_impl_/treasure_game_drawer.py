# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:51:33 2014

@author: gdk
"""
import math
import random
from collections import defaultdict

import pygame
from pygame.locals import *

import os

from gym_multi_treasure_game.envs.treasure_game_impl_._objects import Handle, Door, Key, GoldCoin, Bolt
from gym_multi_treasure_game.envs.treasure_game_impl_._constants import X_SCALE, Y_SCALE, OPEN_SPACE, WALL, LADDER, \
    AGENT_WALL, AGENT_OPEN_SPACE, AGENT_DOOR_OPEN, AGENT_DOOR_CLOSED, AGENT_GOLD, AGENT_LADDER, AGENT_BOLT_LOCKED, \
    AGENT_BOLT_UNLOCKED, AGENT_KEY, AGENT_HANDLE_UP, AGENT_HANDLE_DOWN, AGENT_NORMALISE_CONSTANT, BACKGROUND_SPRITE, \
    WALL_SPRITE, LADDER_SPRITE, DOOR_CLOSED_SPRITE, DOOR_OPEN_SPRITE, KEY_SPRITE, COIN_SPRITE, BOLT_OPEN_SPRITE, \
    BOLT_CLOSED_SPRITE, HERO_SPRITE, HANDLE_BASE_SPRITE, HANDLE_SHAFT_SPRITE, FLOOR_SPRITE, TORCH, TORCH_SPRITE, BANNER, \
    BANNER_SPRITE, MARKER, MARKER_SPRITE
import numpy as np
from gym_multi_treasure_game.envs.treasure_game_impl_.treasure_game_impl import TreasureGameImpl_

base_dir = os.path.dirname(os.path.realpath(__file__))


class TreasureGameDrawer_:

    def __init__(self, md: TreasureGameImpl_, display_screen=False, fancy_graphics=False, render_bg=True):

        self.env = md

        if display_screen:
            self.screen = pygame.display.set_mode((self.env.width, self.env.height), DOUBLEBUF)
        else:
            pygame.display.set_mode((1, 1))
            self.screen = pygame.Surface((self.env.width,
                                          self.env.height)).convert()  # we'll use gym to render. So just use a surface as the screen!

        self.images = self.load_sprites()
        self.random_generator = random.Random()
        self.random_generator.seed(1)
        self.random_images = self.load_random_images()
        self.seed = 12
        self.fancy_graphics = fancy_graphics
        self.render_bg = render_bg

    # make nice by randomising background

    def load_random_images(self):
        images = defaultdict(list)
        for i in range(5):
            background = pygame.transform.scale(
                pygame.image.load(base_dir + '/sprites/background/background_{}.png'.format(i)).convert_alpha(),
                (X_SCALE, Y_SCALE))
            images[BACKGROUND_SPRITE].append(background)

            wallpic = pygame.transform.scale(
                pygame.image.load(base_dir + '/sprites/wall/wall_{}.png'.format(i)).convert_alpha(),
                (X_SCALE, Y_SCALE))

            images[WALL_SPRITE].append(wallpic)

            floorpic = pygame.transform.scale(
                pygame.image.load(base_dir + '/sprites/floor/floor-{}.png'.format(i)).convert_alpha(),
                (X_SCALE, Y_SCALE))
            images[FLOOR_SPRITE].append(floorpic)
            torchpic = pygame.transform.scale(
                pygame.image.load(base_dir + '/sprites/torch/torch_{}.png'.format(i)).convert_alpha(),
                (X_SCALE, Y_SCALE))
            images[TORCH_SPRITE].append(torchpic)
        return images

    def _get_random_sprite(self, key):
        return self.random_generator.choice(self.random_images[key])

    def load_sprites(self):
        images = dict()

        ladder = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/ladder.png').convert_alpha(),
                                        (X_SCALE, Y_SCALE))
        images[LADDER_SPRITE] = ladder

        background = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/background.png').convert_alpha(),
                                            (X_SCALE, Y_SCALE))
        images[BACKGROUND_SPRITE] = background

        wallpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/wall.png').convert_alpha(),
                                         (X_SCALE, Y_SCALE))
        images[WALL_SPRITE] = wallpic

        torchpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/torch.png').convert_alpha(),
                                          (X_SCALE, Y_SCALE))
        images[TORCH_SPRITE] = torchpic

        bannerpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/banner.png').convert_alpha(),
                                           (X_SCALE, Y_SCALE))
        images[BANNER_SPRITE] = bannerpic

        markerpic = pygame.transform.scale(
            pygame.image.load(base_dir + '/sprites/background/stone_black_marked4.png').convert_alpha(),
            (X_SCALE, Y_SCALE))
        images[MARKER_SPRITE] = markerpic

        goldpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/gold.png').convert_alpha(),
                                         (X_SCALE, Y_SCALE))
        images[COIN_SPRITE] = goldpic

        keypic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/key.png').convert_alpha(),
                                        (X_SCALE, Y_SCALE))
        images[KEY_SPRITE] = keypic

        heropic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/hero.png').convert_alpha(),
                                         (X_SCALE, Y_SCALE))
        images[HERO_SPRITE] = heropic

        opendoorpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/open-door.png').convert_alpha(),
                                             (X_SCALE, Y_SCALE))
        images[DOOR_OPEN_SPRITE] = opendoorpic

        closeddoorpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/closeddoor.png').convert_alpha(),
                                               (X_SCALE, Y_SCALE))
        images[DOOR_CLOSED_SPRITE] = closeddoorpic

        openboltpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/bolt-open.png').convert_alpha(),
                                             (X_SCALE, Y_SCALE))
        images[BOLT_OPEN_SPRITE] = openboltpic

        closedboltpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/bolt-locked.png').convert_alpha(),
                                               (X_SCALE, Y_SCALE))
        images[BOLT_CLOSED_SPRITE] = closedboltpic

        handlepic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/handle-base.png').convert_alpha(),
                                           (X_SCALE, Y_SCALE))
        images[HANDLE_BASE_SPRITE] = handlepic

        handleshaftpic = pygame.transform.scale(
            pygame.image.load(base_dir + '/sprites/handle-shaft.png').convert_alpha(), (X_SCALE // 2, Y_SCALE))
        images[HANDLE_SHAFT_SPRITE] = handleshaftpic

        return images

    def draw_domain(self, show_screen=True, ):

        self.random_generator.seed(self.seed)
        self.screen.fill((0, 0, 0))

        for i in range(0, self.env.cell_height):
            for j in range(0, self.env.cell_width):
                if (self.env.description[i][j] == WALL):

                    key = WALL_SPRITE
                    if i > 0 and self.env.description[i - 1][j] != WALL:
                        key = FLOOR_SPRITE

                    self.screen.blit(self._get_random_sprite(key), (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == LADDER):
                    self.screen.blit(self.images[LADDER_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == OPEN_SPACE):
                    if self.render_bg:
                        self.screen.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == TORCH):
                    if self.render_bg:
                        self.screen.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                    self.screen.blit(self._get_random_sprite(TORCH_SPRITE), (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == BANNER):
                    if self.render_bg:
                        self.screen.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                    self.screen.blit(self.images[BANNER_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == MARKER):
                    if self.render_bg:
                        self.screen.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                    self.screen.blit(self.images[MARKER_SPRITE], (j * X_SCALE, i * Y_SCALE))

        if self.env.facing_right or not self.fancy_graphics:
            self.screen.blit(self.images[HERO_SPRITE], (self.env.playerx - X_SCALE / 2, self.env.playery))
        else:
            self.screen.blit(pygame.transform.flip(self.images[HERO_SPRITE], True, False),
                             (self.env.playerx - X_SCALE / 2, self.env.playery))
        # draw objects in front of agent for PCA purposes
        for obj in self.env.objects:
            self.draw_object(obj, self.screen)

        if show_screen:
            pygame.display.flip()

    def draw_background_to_surface(self):

        self.random_generator.seed(self.seed)
        draw_surf = pygame.Surface((self.env.width, self.env.height))
        draw_surf.fill((0, 0, 0))

        for i in range(0, self.env.cell_height):
            for j in range(0, self.env.cell_width):

                if (self.env.description[i][j] == WALL):
                    key = WALL_SPRITE
                    if i > 0 and self.env.description[i - 1][j] != WALL:
                        key = FLOOR_SPRITE
                    draw_surf.blit(self._get_random_sprite(key), (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == LADDER):
                    draw_surf.blit(self.images[LADDER_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == OPEN_SPACE):
                    if self.render_bg:
                        draw_surf.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == TORCH):
                    if self.render_bg:
                        draw_surf.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                    draw_surf.blit(self.images[TORCH_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == BANNER):
                    if self.render_bg:
                        draw_surf.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                    draw_surf.blit(self.images[BANNER_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif (self.env.description[i][j] == MARKER):
                    if self.render_bg:
                        draw_surf.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))
                    draw_surf.blit(self.images[MARKER_SPRITE], (j * X_SCALE, i * Y_SCALE))

        return draw_surf

    def draw_to_surface(self):

        draw_surf = self.draw_background_to_surface()

        for obj in self.env.objects:
            self.draw_object(obj, draw_surf)

        if self.env.facing_right or not self.fancy_graphics:
            draw_surf.blit(self.images[HERO_SPRITE], (self.env.playerx - X_SCALE / 2, self.env.playery))
        else:
            draw_surf.blit(pygame.transform.flip(self.images[HERO_SPRITE], True, False),
                           (self.env.playerx - X_SCALE / 2, self.env.playery))
        return draw_surf

    def blit_alpha(self, target, source, location, opacity):
        x = location[0]
        y = location[1]
        temp = pygame.Surface((source.get_width(), source.get_height())).convert()
        temp.blit(target, (-x, -y))
        temp.blit(source, (0, 0))
        temp.set_alpha(opacity)
        target.blit(temp, location)

    def blend(self, surf, alpha_objs, alpha_player):

        new_surf = pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA, 32)
        new_surf.convert_alpha()
        for obj in self.env.objects:
            if (isinstance(obj, Handle)):
                angle = ((math.pi / 2.0) * obj.get_angle()) + math.pi / 4.0
                r = Y_SCALE * 0.75
                startpos = (obj.x + X_SCALE / 2, obj.y + Y_SCALE)
                endpos = (int(startpos[0] + (r * math.cos(angle))), int(startpos[1] - (r * math.sin(angle))))

                pygame.draw.line(new_surf, (47, 79, 79), startpos, endpos, 5)
                pygame.draw.circle(new_surf, (255, 0, 0), endpos, int(X_SCALE / 10), 0)
                surf.blit(self.images[HANDLE_BASE_SPRITE], (obj.x, obj.y))
            else:
                self.draw_object(obj, new_surf)
        self.blit_alpha(surf, new_surf, (0, 0), int(255 * alpha_objs))

        if self.env.facing_right or not self.fancy_graphics:
            self.blit_alpha(surf, self.images[HERO_SPRITE], (self.env.playerx - X_SCALE / 2, self.env.playery),
                            int(255 * alpha_player))
        else:
            self.blit_alpha(surf, pygame.transform.flip(self.images[HERO_SPRITE], True, False),
                            (self.env.playerx - X_SCALE / 2, self.env.playery),
                            int(255 * alpha_player))

    def draw_to_file(self, fname):

        draw_surf = self.draw_to_surface()
        pygame.image.save(draw_surf, fname)

    def draw_object(self, obj, surf):

        if (obj.x < 0):
            return

        if (isinstance(obj, Door)):
            img = self.images[DOOR_OPEN_SPRITE]
            if (obj.door_closed()):
                img = self.images[DOOR_CLOSED_SPRITE]
            surf.blit(img, (obj.x, obj.y))
        elif (isinstance(obj, Key)):
            surf.blit(self.images[KEY_SPRITE], (obj.x, obj.y))
        elif (isinstance(obj, GoldCoin)):
            surf.blit(self.images[COIN_SPRITE], (obj.x, obj.y))
        elif (isinstance(obj, Bolt)):
            img = self.images[BOLT_OPEN_SPRITE]
            if (obj.get_locked()):
                img = self.images[BOLT_CLOSED_SPRITE]
            surf.blit(img, (obj.x, obj.y))
        elif (isinstance(obj, Handle)):
            angle = ((math.pi / 2.0) * obj.get_angle()) + math.pi / 4.0
            r = Y_SCALE * 0.75

            startpos = (obj.x + X_SCALE / 2, obj.y + Y_SCALE)
            endpos = (int(startpos[0] + (r * math.cos(angle))), int(startpos[1] - (r * math.sin(angle))))

            pygame.draw.line(surf, (47, 79, 79), startpos, endpos, 5)
            pygame.draw.circle(surf, (255, 0, 0), endpos, int(X_SCALE / 10), 0)
            surf.blit(self.images[HANDLE_BASE_SPRITE], (obj.x, obj.y))
        else:
            print("unknown object during draw")
            print(type(obj).__name__)

    def draw_local_view(self, state=None, split=False):

        if state is None:
            x, y = self.env.playerx, self.env.playery
        else:
            x, y = state[0] * self.env.width, state[1] * self.env.height

        N = 3
        size_x = X_SCALE * N
        size_y = Y_SCALE * N

        if split:
            surface = pygame.Surface((size_x, size_y))
            surface2 = pygame.Surface((size_x, Y_SCALE))
            surface2.fill((0, 0, 0))
        else:
            surface = pygame.Surface((size_x, size_y + Y_SCALE))
        surface.fill((0, 0, 0))

        left = max(0, x - X_SCALE // 2 - X_SCALE)
        top = max(0, y - Y_SCALE)
        left = min(self.screen.get_width() - size_x, left)
        top = min(self.screen.get_height() - size_y, top)

        # draw local view
        surface.blit(self.screen.subsurface(left, top, size_x, size_y), (0, 0))

        # draw inventory
        bag = [int(self.env.player_got_key()), int(self.env.player_got_goldcoin())]
        has_gold = False
        if not np.isnan(bag[1]):
            if int(round(bag[1])) == 1:
                has_gold = True
                if split:
                    surface2.blit(self.images[COIN_SPRITE], (0, 0))
                else:
                    surface.blit(self.images[COIN_SPRITE], (0, N * Y_SCALE))
        if not has_gold:
            if not np.isnan(bag[0]):
                if int(round(bag[0])) == 1:

                    if split:
                        surface2.blit(self.images[KEY_SPRITE], (0, 0))
                        if self.env.key_dropped():
                            pygame.draw.line(surface2, (250, 182, 27), (0, 0),
                                             (X_SCALE, Y_SCALE), 5)
                            pygame.draw.line(surface2, (250, 182, 27), (X_SCALE, N * Y_SCALE),
                                             (0, Y_SCALE), 5)
                    else:
                        surface.blit(self.images[KEY_SPRITE], (0, N * Y_SCALE))
                        if self.env.key_dropped():
                            pygame.draw.line(surface, (250, 182, 27), (0, N * Y_SCALE),
                                             (X_SCALE, N * Y_SCALE + Y_SCALE), 5)
                            pygame.draw.line(surface, (250, 182, 27), (X_SCALE, N * Y_SCALE),
                                             (0, N * Y_SCALE + Y_SCALE), 5)



                    # if int(round(bag[1])) == 0:
            #     pygame.draw.line(surface, (255, 0, 0), (X_SCALE, N * Y_SCALE),
            #                      (X_SCALE * 2, N * Y_SCALE + Y_SCALE), 2)
            #     pygame.draw.line(surface, (255, 0, 0), (2 * X_SCALE, N * Y_SCALE),
            #                      (X_SCALE, N * Y_SCALE + Y_SCALE), 2)

        # if bag == 2:
        #     draw_surf.blit(self.images[COIN_SPRITE], (0, N * Y_SCALE))
        #     self.screen.blit(self.images[COIN_SPRITE], (0, N * Y_SCALE))
        # elif bag == 1:
        #     draw_surf.blit(self.images[KEY_SPRITE], (0, N * Y_SCALE))
        #     self.screen.blit(self.images[KEY_SPRITE], (0, N * Y_SCALE))
        # pygame.display.flip()

        if split:
            return surface, surface2
        return surface
