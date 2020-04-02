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

from gym_multi_treasure_game.envs._treasure_game_impl._objects import Handle, Door, Key, GoldCoin, Bolt
from gym_multi_treasure_game.envs._treasure_game_impl._constants import X_SCALE, Y_SCALE, OPEN_SPACE, WALL, LADDER, \
    AGENT_WALL, AGENT_OPEN_SPACE, AGENT_DOOR_OPEN, AGENT_DOOR_CLOSED, AGENT_GOLD, AGENT_LADDER, AGENT_BOLT_LOCKED, \
    AGENT_BOLT_UNLOCKED, AGENT_KEY, AGENT_HANDLE_UP, AGENT_HANDLE_DOWN, AGENT_NORMALISE_CONSTANT, BACKGROUND_SPRITE, \
    WALL_SPRITE, LADDER_SPRITE, DOOR_CLOSED_SPRITE, DOOR_OPEN_SPRITE, KEY_SPRITE, COIN_SPRITE, BOLT_OPEN_SPRITE, \
    BOLT_CLOSED_SPRITE, HERO_SPRITE, HANDLE_BASE_SPRITE, HANDLE_SHAFT_SPRITE, FLOOR_SPRITE
import numpy as np
from gym_multi_treasure_game.envs._treasure_game_impl._treasure_game_impl import _TreasureGameImpl

base_dir = os.path.dirname(os.path.realpath(__file__))


class _TreasureGameDrawer:

    def __init__(self, md: _TreasureGameImpl, display_screen=False):

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
        return images

    def _get_random_sprite(self, key):
        # if key == BACKGROUND_SPRITE:
        #     return self.random_images[key][0]
        return self.random_generator.choice(self.random_images[key])

    def load_sprites(self):
        images = [None] * 12

        ladder = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/ladder.png').convert_alpha(),
                                        (X_SCALE, Y_SCALE))
        images[LADDER_SPRITE] = ladder

        background = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/background.png').convert_alpha(),
                                            (X_SCALE, Y_SCALE))
        images[BACKGROUND_SPRITE] = background

        wallpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/wall.png').convert_alpha(),
                                         (X_SCALE, Y_SCALE))
        images[WALL_SPRITE] = wallpic

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
                    self.screen.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))

        for obj in self.env.objects:
            self.draw_object(obj, self.screen)

        if self.env.facing_right:
            self.screen.blit(self.images[HERO_SPRITE], (self.env.playerx - X_SCALE / 2, self.env.playery))
        else:
            self.screen.blit(pygame.transform.flip(self.images[HERO_SPRITE], True, False),
                             (self.env.playerx - X_SCALE / 2, self.env.playery))
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
                    draw_surf.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * X_SCALE, i * Y_SCALE))

        return draw_surf

    def draw_to_surface(self):

        draw_surf = self.draw_background_to_surface()

        for obj in self.env.objects:
            self.draw_object(obj, draw_surf)

        if self.env.facing_right:
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

        if self.env.facing_right:
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

    def draw_local_view(self, state=None):

        if state is None:
            state = self.env.current_observation()

        recent_tiles = [0]

        bag = state[-2:]
        state = state[:-2]
        state *= AGENT_NORMALISE_CONSTANT

        n = int(np.sqrt(len(state)))
        size_x = X_SCALE * n
        size_y = Y_SCALE * (n + 1)
        blocks = state.reshape((n, n))
        draw_surf = pygame.Surface((size_x, size_y))
        draw_surf.fill((0, 0, 0))
        # self.screen.fill((0, 0, 0))
        mid = n // 2
        for i in range(0, len(blocks)):
            for j in range(0, len(blocks[i])):

                if np.isnan(blocks[i][j]):
                    if i == mid and j == mid:
                        draw_surf.blit(self.images[HERO_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    continue

                x = int(round(blocks[i][j]))

                # always have background in the background unless it's the centre
                if i != 1 or j != 1:
                    img = self._get_random_sprite(BACKGROUND_SPRITE)
                    draw_surf.blit(img, (j * X_SCALE, i * Y_SCALE))

                if x == AGENT_WALL:
                    key = WALL_SPRITE
                    if i > 0 and blocks[i - 1][j] != AGENT_WALL:
                        key = FLOOR_SPRITE

                    img = self._get_random_sprite(key)
                    draw_surf.blit(img, (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(img, (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_OPEN_SPACE:
                    img = self._get_random_sprite(BACKGROUND_SPRITE)
                    draw_surf.blit(img, (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(img, (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_DOOR_OPEN:
                    draw_surf.blit(self.images[DOOR_OPEN_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(self.images[DOOR_OPEN_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_DOOR_CLOSED:
                    draw_surf.blit(self.images[DOOR_CLOSED_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(self.images[DOOR_CLOSED_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_GOLD:
                    draw_surf.blit(self.images[COIN_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(self.images[COIN_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_LADDER:
                    draw_surf.blit(self.images[LADDER_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(self.images[LADDER_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_BOLT_LOCKED:
                    draw_surf.blit(self.images[BOLT_CLOSED_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(self.images[BOLT_CLOSED_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_BOLT_UNLOCKED:
                    draw_surf.blit(self.images[BOLT_OPEN_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(self.images[BOLT_OPEN_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_KEY:
                    draw_surf.blit(self.images[KEY_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    # self.screen.blit(self.images[KEY_SPRITE], (j * X_SCALE, i * Y_SCALE))
                elif x == AGENT_HANDLE_UP or x == AGENT_HANDLE_DOWN:

                    im = self.images[HANDLE_SHAFT_SPRITE]
                    angleRaw = random.uniform(0.85, 1.0) if AGENT_HANDLE_UP == x else random.uniform(0, 0.15)
                    angle = ((math.pi / 2.0) * angleRaw) - math.pi / 4.0

                    im = pygame.transform.rotate(im, angle * 180.0 / math.pi)
                    # This rotation draws a bounding box around the resulting image, and then
                    # translates the results so that the bounding box is aligned with the top-
                    # left pixel of the original image. To figure out how much that moves the
                    # center by, we do the following calculation:
                    dx = X_SCALE * math.fabs(math.cos(angle) + math.sin(math.fabs(angle)) - 1)
                    dy = Y_SCALE * math.fabs(math.cos(angle) + math.sin(math.fabs(angle)) - 1)
                    x = j * X_SCALE
                    y = i * Y_SCALE
                    handle_pivot = ((x - X_SCALE / 2) - dx, (y - 5) - dy)
                    draw_surf.blit(self.images[HANDLE_BASE_SPRITE], (x, y))
                    # draw_surf.blit(im, handle_pivot)
                    draw_surf.blit(im, (x, y))

                #     self.screen.blit(self.images[HANDLE_BASE_SPRITE], (x, y))
                #     self.screen.blit(im, handle_pivot)
                #
                if i == mid and j == mid:
                    # self.screen.blit(self.images[HERO_SPRITE], (j * X_SCALE, i * Y_SCALE))
                    draw_surf.blit(self.images[HERO_SPRITE], (j * X_SCALE, i * Y_SCALE))

        if not np.isnan(bag[0]):
            draw_surf.blit(self.images[KEY_SPRITE], (0, len(blocks) * Y_SCALE))
            if int(round(bag[0])) == 0:
                pygame.draw.line(draw_surf, (255, 0, 0), (0, len(blocks) * Y_SCALE),
                                 (X_SCALE, len(blocks) * Y_SCALE + Y_SCALE), 2)
                pygame.draw.line(draw_surf, (255, 0, 0), (X_SCALE, len(blocks) * Y_SCALE),
                                 (0, len(blocks) * Y_SCALE + Y_SCALE), 2)

        if not np.isnan(bag[1]):
            draw_surf.blit(self.images[COIN_SPRITE], (X_SCALE, len(blocks) * Y_SCALE))
            if int(round(bag[1])) == 0:
                pygame.draw.line(draw_surf, (255, 0, 0), (X_SCALE, len(blocks) * Y_SCALE),
                                 (X_SCALE * 2, len(blocks) * Y_SCALE + Y_SCALE), 2)
                pygame.draw.line(draw_surf, (255, 0, 0), (2 * X_SCALE, len(blocks) * Y_SCALE),
                                 (X_SCALE, len(blocks) * Y_SCALE + Y_SCALE), 2)

        # if bag == 2:
        #     draw_surf.blit(self.images[COIN_SPRITE], (0, len(blocks) * Y_SCALE))
        #     self.screen.blit(self.images[COIN_SPRITE], (0, len(blocks) * Y_SCALE))
        # elif bag == 1:
        #     draw_surf.blit(self.images[KEY_SPRITE], (0, len(blocks) * Y_SCALE))
        #     self.screen.blit(self.images[KEY_SPRITE], (0, len(blocks) * Y_SCALE))
        # pygame.display.flip()
        return draw_surf
