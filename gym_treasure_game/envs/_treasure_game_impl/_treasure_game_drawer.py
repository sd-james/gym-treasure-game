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

from gym_treasure_game.envs._treasure_game_impl._cell_types import WALL, LADDER, OPEN_SPACE
from gym_treasure_game.envs._treasure_game_impl._objects import handle, door, key, goldcoin, bolt
from gym_treasure_game.envs._treasure_game_impl._scale import xscale, yscale

base_dir = os.path.dirname(os.path.realpath(__file__))

BACKGROUND_SPRITE = 0
WALL_SPRITE = 1
LADDER_SPRITE = 2
DOOR_CLOSED_SPRITE = 3
DOOR_OPEN_SPRITE = 4
KEY_SPRITE = 5
COIN_SPRITE = 6
BOLT_OPEN_SPRITE = 7
BOLT_CLOSED_SPRITE = 8
HERO_SPRITE = 9
HANDLE_SPRITE = 10
HANDLE_SHAFT_SPRITE = 11
FLOOR_SPRITE = 12


class _TreasureGameDrawer:

    def __init__(self, md, display_screen=False):

        self.md = md

        if display_screen:
            self.screen = pygame.display.set_mode((self.md.width, self.md.height), DOUBLEBUF)
        else:
            pygame.display.set_mode((1, 1))
            self.screen = pygame.Surface((self.md.width,
                                          self.md.height)).convert()  # we'll use gym to render. So just use a surface as the screen!
        #

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
                (xscale, yscale))
            images[BACKGROUND_SPRITE].append(background)

            wallpic = pygame.transform.scale(
                pygame.image.load(base_dir + '/sprites/wall/wall_{}.png'.format(i)).convert_alpha(),
                (xscale, yscale))
            images[WALL_SPRITE].append(wallpic)

            floorpic = pygame.transform.scale(
                pygame.image.load(base_dir + '/sprites/floor/floor-{}.png'.format(i)).convert_alpha(),
                (xscale, yscale))
            images[FLOOR_SPRITE].append(floorpic)
        return images

    def _get_random_sprite(self, key):
        # if key == BACKGROUND_SPRITE:
        #     return self.random_images[key][0]
        return self.random_generator.choice(self.random_images[key])

    def load_sprites(self):
        images = [None] * 12

        ladder = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/ladder.png').convert_alpha(),
                                        (xscale, yscale))
        images[LADDER_SPRITE] = ladder

        background = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/background.png').convert_alpha(),
                                            (xscale, yscale))
        images[BACKGROUND_SPRITE] = background

        wallpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/wall.png').convert_alpha(),
                                         (xscale, yscale))
        images[WALL_SPRITE] = wallpic

        goldpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/gold.png').convert_alpha(),
                                         (xscale, yscale))
        images[COIN_SPRITE] = goldpic

        keypic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/key.png').convert_alpha(),
                                        (xscale, yscale))
        images[KEY_SPRITE] = keypic

        heropic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/hero.png').convert_alpha(),
                                         (xscale, yscale))
        images[HERO_SPRITE] = heropic

        opendoorpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/open-door.png').convert_alpha(),
                                             (xscale, yscale))
        images[DOOR_OPEN_SPRITE] = opendoorpic

        closeddoorpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/closeddoor.png').convert_alpha(),
                                               (xscale, yscale))
        images[DOOR_CLOSED_SPRITE] = closeddoorpic

        openboltpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/bolt-open.png').convert_alpha(),
                                             (xscale, yscale))
        images[BOLT_OPEN_SPRITE] = openboltpic

        closedboltpic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/bolt-locked.png').convert_alpha(),
                                               (xscale, yscale))
        images[BOLT_CLOSED_SPRITE] = closedboltpic

        handlepic = pygame.transform.scale(pygame.image.load(base_dir + '/sprites/handle-base.png').convert_alpha(),
                                           (xscale, yscale))
        images[HANDLE_SPRITE] = handlepic

        handleshaftpic = pygame.transform.scale(
            pygame.image.load(base_dir + '/sprites/handle-shaft.png').convert_alpha(), (xscale // 2, yscale))
        images[HANDLE_SHAFT_SPRITE] = handleshaftpic

        return images

    def draw_domain(self, show_screen=True):
        self.random_generator.seed(self.seed)
        self.screen.fill((0, 0, 0))

        for i in range(0, self.md.cell_height):
            for j in range(0, self.md.cell_width):
                if (self.md.description[i][j] == WALL):

                    key = WALL_SPRITE
                    if i > 0 and self.md.description[i - 1][j] != WALL:
                        key = FLOOR_SPRITE

                    self.screen.blit(self._get_random_sprite(key), (j * xscale, i * yscale))
                elif (self.md.description[i][j] == LADDER):
                    self.screen.blit(self.images[LADDER_SPRITE], (j * xscale, i * yscale))
                elif (self.md.description[i][j] == OPEN_SPACE):
                    self.screen.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * xscale, i * yscale))

        for obj in self.md.objects:
            self.draw_object(obj, self.screen)

        if self.md.facing_right:
            self.screen.blit(self.images[HERO_SPRITE], (self.md.playerx - xscale / 2, self.md.playery))
        else:
            self.screen.blit(pygame.transform.flip(self.images[HERO_SPRITE], True, False),
                             (self.md.playerx - xscale / 2, self.md.playery))
        if show_screen:
            pygame.display.flip()

    def draw_background_to_surface(self):
        self.random_generator.seed(self.seed)
        draw_surf = pygame.Surface((self.md.width, self.md.height))
        draw_surf.fill((0, 0, 0))

        for i in range(0, self.md.cell_height):
            for j in range(0, self.md.cell_width):
                if (self.md.description[i][j] == WALL):
                    key = WALL_SPRITE
                    if i > 0 and self.md.description[i - 1][j] != WALL:
                        key = FLOOR_SPRITE
                    draw_surf.blit(self._get_random_sprite(key), (j * xscale, i * yscale))
                elif (self.md.description[i][j] == LADDER):
                    draw_surf.blit(self.images[LADDER_SPRITE], (j * xscale, i * yscale))
                elif (self.md.description[i][j] == OPEN_SPACE):
                    draw_surf.blit(self._get_random_sprite(BACKGROUND_SPRITE), (j * xscale, i * yscale))

        return draw_surf

    def draw_to_surface(self):

        draw_surf = self.draw_background_to_surface()

        for obj in self.md.objects:
            self.draw_object(obj, draw_surf)

        if self.md.facing_right:
            draw_surf.blit(self.images[HERO_SPRITE], (self.md.playerx - xscale / 2, self.md.playery))
        else:
            draw_surf.blit(pygame.transform.flip(self.images[HERO_SPRITE], True, False),
                           (self.md.playerx - xscale / 2, self.md.playery))
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

        new_surf = pygame.Surface((self.md.width, self.md.height), pygame.SRCALPHA, 32)
        new_surf.convert_alpha()
        for obj in self.md.objects:
            if (isinstance(obj, handle)):
                angle = ((math.pi / 2.0) * obj.get_angle()) + math.pi / 4.0
                r = yscale * 0.75
                startpos = (obj.x + xscale / 2, obj.y + yscale)
                endpos = (int(startpos[0] + (r * math.cos(angle))), int(startpos[1] - (r * math.sin(angle))))

                pygame.draw.line(new_surf, (47, 79, 79), startpos, endpos, 5)
                pygame.draw.circle(new_surf, (255, 0, 0), endpos, int(xscale / 10), 0)
                surf.blit(self.images[HANDLE_SPRITE], (obj.x, obj.y))
            else:
                self.draw_object(obj, new_surf)
        self.blit_alpha(surf, new_surf, (0, 0), int(255 * alpha_objs))

        if self.md.facing_right:
            self.blit_alpha(surf, self.images[HERO_SPRITE], (self.md.playerx - xscale / 2, self.md.playery),
                            int(255 * alpha_player))
        else:
            self.blit_alpha(surf, pygame.transform.flip(self.images[HERO_SPRITE], True, False),
                            (self.md.playerx - xscale / 2, self.md.playery),
                            int(255 * alpha_player))

    def draw_to_file(self, fname):

        draw_surf = self.draw_to_surface()
        pygame.image.save(draw_surf, fname)

    def draw_object(self, obj, surf):

        if (obj.x < 0):
            return

        if (isinstance(obj, door)):
            img = self.images[DOOR_OPEN_SPRITE]
            if (obj.door_closed()):
                img = self.images[DOOR_CLOSED_SPRITE]
            surf.blit(img, (obj.x, obj.y))
        elif (isinstance(obj, key)):
            surf.blit(self.images[KEY_SPRITE], (obj.x, obj.y))
        elif (isinstance(obj, goldcoin)):
            surf.blit(self.images[COIN_SPRITE], (obj.x, obj.y))
        elif (isinstance(obj, bolt)):
            img = self.images[BOLT_OPEN_SPRITE]
            if (obj.get_locked()):
                img = self.images[BOLT_CLOSED_SPRITE]
            surf.blit(img, (obj.x, obj.y))
        elif (isinstance(obj, handle)):
            angle = ((math.pi / 2.0) * obj.get_angle()) + math.pi / 4.0
            r = yscale * 0.75

            startpos = (obj.x + xscale / 2, obj.y + yscale)
            endpos = (int(startpos[0] + (r * math.cos(angle))), int(startpos[1] - (r * math.sin(angle))))

            pygame.draw.line(surf, (47, 79, 79), startpos, endpos, 5)
            pygame.draw.circle(surf, (255, 0, 0), endpos, int(xscale / 10), 0)
            surf.blit(self.images[HANDLE_SPRITE], (obj.x, obj.y))
        else:
            print("unknown object during draw")
            print(type(obj).__name__)
