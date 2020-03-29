# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:35:46 2014

@author: gdk
"""

from gym_multi_treasure_game.envs._treasure_game_impl._option import _Option
from gym_multi_treasure_game.envs._treasure_game_impl._objects import Handle, Bolt
from gym_multi_treasure_game.envs._treasure_game_impl._constants import X_SCALE, Y_SCALE, OPEN_SPACE, WALL, LADDER, \
    ACTION_NOP, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP, ACTION_INTERACT


class go_left_option(_Option):

    def __init__(self, md, drawer=None):
        super().__init__(md, drawer)
        self.start_cell = None
        self.target_cell = None

    def can_run(self):
        pcell = self.md.get_player_cell()
        tcell = self.get_target_cell(pcell)

        if (tcell == None):
            return False

        xc = pcell[0]
        yc = pcell[1]
        tc = tcell[0]

        while (xc >= tc):
            if (self.md.object_type_at_cell(xc, yc) != OPEN_SPACE):
                return False
            if (self.md.object_type_at_cell(xc, yc + 1) == OPEN_SPACE):
                return False
            xc = xc - 1

        return True

    def get_target_cell(self, tcell):
        xc = tcell[0] - 1
        yc = tcell[1]

        while (not self.is_target_cell(xc, yc)):
            xc = xc - 1
            if (xc < 0):
                return None

        return (xc, yc)

    def is_target_cell(self, xc, yc):

        if (self.md.object_type_at_cell(xc, yc - 1) == LADDER):
            return True
        if (self.md.object_type_at_cell(xc, yc + 1) == LADDER):
            return True
        if (self.md.object_type_at_cell(xc - 1, yc) == WALL):
            return True
        if self.md.is_object_at(xc, yc) or self.md.is_closed_door_at(xc - 1, yc):
            return True
        if (self.md.object_type_at_cell(xc - 1, yc + 1) == OPEN_SPACE):
            return True

        return False

    def close_enough_to(self, tcell):
        tx = (tcell[0] * X_SCALE) + (X_SCALE / 2)
        diff = abs(tx - self.md.playerx)
        return (diff < self.md.x_incr)

    def policy_step(self):

        if ((self.start_cell == None) or (self.target_cell == None)):
            self.start_cell = self.md.get_player_cell()
            self.target_cell = self.get_target_cell(self.start_cell)

        if (self.close_enough_to(self.target_cell)):
            self.done = True
            self.start_cell = None
            self.target_cell = None

        return ACTION_LEFT


class go_right_option(_Option):

    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)
        self.start_cell = None
        self.target_cell = None

    def can_run(self):
        pcell = self.md.get_player_cell()
        tcell = self.get_target_cell(pcell)

        if (tcell == None):
            return False

        xc = pcell[0]
        yc = pcell[1]
        tc = tcell[0]

        while (xc <= tc):
            if (self.md.object_type_at_cell(xc, yc) != OPEN_SPACE):
                return False
            if (self.md.object_type_at_cell(xc, yc + 1) == OPEN_SPACE):
                return False
            xc = xc + 1

        return True

    def get_target_cell(self, tcell):
        xc = tcell[0] + 1
        yc = tcell[1]

        while (not self.is_target_cell(xc, yc)):
            xc = xc + 1
            if (xc < 0):
                return None

        return (xc, yc)

    def is_target_cell(self, xc, yc):

        if (self.md.object_type_at_cell(xc, yc - 1) == LADDER):
            return True
        if (self.md.object_type_at_cell(xc, yc + 1) == LADDER):
            return True
        if (self.md.object_type_at_cell(xc + 1, yc) == WALL):
            return True
        if self.md.is_object_at(xc, yc) or self.md.is_closed_door_at(xc + 1, yc):
            return True
        if (self.md.object_type_at_cell(xc + 1, yc + 1) == OPEN_SPACE):
            return True

        return False

    def close_enough_to(self, tcell):
        tx = (tcell[0] * X_SCALE) + (X_SCALE / 2)
        diff = abs(tx - self.md.playerx)
        return (diff < self.md.x_incr)

    def policy_step(self):

        if ((self.start_cell == None) or (self.target_cell == None)):
            self.start_cell = self.md.get_player_cell()
            self.target_cell = self.get_target_cell(self.start_cell)

        if (self.close_enough_to(self.target_cell)):
            self.done = True
            self.start_cell = None
            self.target_cell = None

        return ACTION_RIGHT


class up_ladder_option(_Option):

    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)

    def can_run(self):
        return self.md.can_go_up()

    def policy_step(self):
        if (not self.md.can_go_up()):
            self.done = True
            return ACTION_NOP

        return ACTION_UP


class down_ladder_option(_Option):

    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)

    def can_run(self):
        return self.md.can_go_down()

    def policy_step(self):
        if (not self.md.can_go_down()):
            self.done = True
            return ACTION_NOP

        return ACTION_DOWN


class down_left_option(_Option):

    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)
        self.start_cell = None
        self.target_cell = None

    def can_run(self):
        pcell = self.md.get_player_cell()
        xc = pcell[0]
        yc = pcell[1]

        if (self.md.object_type_at_cell(xc - 1, yc) != OPEN_SPACE):
            return False
        if (self.md.object_type_at_cell(xc - 1, yc + 1) != OPEN_SPACE):
            return False

        return True

    def get_target_cell(self, tcell):

        xc = tcell[0] - 1
        yc = tcell[1] + 1

        while (self.md.object_type_at_cell(xc, yc) == OPEN_SPACE):
            yc = yc + 1
            if (yc >= self.md.cell_height):
                return None

        return (xc, yc)

    def close_enough_x(self, tcell):
        tx = (tcell[0] * X_SCALE) + (X_SCALE / 2)
        diff = abs(tx - self.md.playerx)
        return (diff < self.md.x_incr)

    def on_floor(self):
        return (not self.md.can_fall())

    def policy_step(self):

        if ((self.start_cell == None) or (self.target_cell == None)):
            self.start_cell = self.md.get_player_cell()
            self.target_cell = self.get_target_cell(self.start_cell)

        if (self.close_enough_x(self.target_cell)):
            if (self.on_floor()):
                self.done = True
                self.start_cell = None
                self.target_cell = None
            return ACTION_NOP

        return ACTION_LEFT


class jump_left_option(_Option):

    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)
        self.start_cell = None
        self.target_cell = None

    def can_run(self):
        pcell = self.md.get_player_cell()
        xc = pcell[0]
        yc = pcell[1]

        if (self.md.object_type_at_cell(xc, yc - 1) != OPEN_SPACE):
            return False
        if (self.md.object_type_at_cell(xc - 1, yc - 1) != OPEN_SPACE):
            return False

        if (not (self.landing(xc - 1, yc - 1) or self.landing(xc - 2, yc - 1))):
            return False

        return True

    def get_target_cell(self, tcell):

        xc = tcell[0]
        yc = tcell[1]

        if (self.landing(xc - 1, yc - 1)):
            return (xc - 1, yc - 1)
        elif (self.landing(xc - 2, yc - 1)):
            return (xc - 2, yc - 1)
        else:
            return None

    def landing(self, xc, yc):
        if (self.md.object_type_at_cell(xc, yc) != OPEN_SPACE):
            return False
        if (self.md.object_type_at_cell(xc, yc + 1) != WALL):
            return False

        return True

    def close_enough_x(self, tcell):
        tx = (tcell[0] * X_SCALE) + (X_SCALE / 2)
        diff = abs(tx - self.md.playerx)
        return (diff < self.md.x_incr)

    def on_floor(self):
        return (not self.md.can_fall())

    def policy_step(self):

        if ((self.start_cell == None) or (self.target_cell == None)):
            self.start_cell = self.md.get_player_cell()
            self.target_cell = self.get_target_cell(self.start_cell)
            return ACTION_JUMP

        if (self.close_enough_x(self.target_cell)):
            if (self.on_floor()):
                self.done = True
                self.start_cell = None
                self.target_cell = None
            return ACTION_NOP

        if ((not self.md.can_fall()) and (not self.md.can_go_left())):
            return ACTION_RIGHT
        else:
            return ACTION_LEFT


class jump_right_option(_Option):

    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)
        self.start_cell = None
        self.target_cell = None

    def can_run(self):
        pcell = self.md.get_player_cell()
        xc = pcell[0]
        yc = pcell[1]

        if (self.md.object_type_at_cell(xc, yc - 1) != OPEN_SPACE):
            return False
        if (self.md.object_type_at_cell(xc + 1, yc - 1) != OPEN_SPACE):
            return False

        if (not (self.landing(xc + 1, yc - 1) or self.landing(xc + 2, yc - 1))):
            return False

        return True

    def get_target_cell(self, tcell):

        xc = tcell[0]
        yc = tcell[1]

        if (self.landing(xc + 1, yc - 1)):
            return (xc + 1, yc - 1)
        elif (self.landing(xc + 2, yc - 1)):
            return (xc + 2, yc - 1)
        else:
            return None

    def landing(self, xc, yc):
        if (self.md.object_type_at_cell(xc, yc) != OPEN_SPACE):
            return False
        if (self.md.object_type_at_cell(xc, yc + 1) != WALL):
            return False

        return True

    def close_enough_x(self, tcell):
        tx = (tcell[0] * X_SCALE) + (X_SCALE / 2)
        diff = abs(tx - self.md.playerx)
        return (diff < self.md.x_incr)

    def on_floor(self):
        return (not self.md.can_fall())

    def policy_step(self):

        if ((self.start_cell == None) or (self.target_cell == None)):
            self.start_cell = self.md.get_player_cell()
            self.target_cell = self.get_target_cell(self.start_cell)
            return ACTION_JUMP

        if (self.close_enough_x(self.target_cell)):
            if (self.on_floor()):
                self.done = True
                self.start_cell = None
                self.target_cell = None
            return ACTION_NOP

        if ((not self.md.can_fall()) and (not self.md.can_go_right())):
            return ACTION_LEFT
        else:
            return ACTION_RIGHT


class down_right_option(_Option):

    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)
        self.start_cell = None
        self.target_cell = None

    def can_run(self):
        pcell = self.md.get_player_cell()
        xc = pcell[0]
        yc = pcell[1]

        if (self.md.object_type_at_cell(xc + 1, yc) != OPEN_SPACE):
            return False
        if (self.md.object_type_at_cell(xc + 1, yc + 1) != OPEN_SPACE):
            return False

        return True

    def get_target_cell(self, tcell):

        xc = tcell[0] + 1
        yc = tcell[1] + 1

        while (self.md.object_type_at_cell(xc, yc) == OPEN_SPACE):
            yc = yc + 1
            if (yc >= self.md.cell_height):
                return None

        return (xc, yc)

    def close_enough_x(self, tcell):
        tx = (tcell[0] * X_SCALE) + (X_SCALE / 2)
        diff = abs(tx - self.md.playerx)
        return (diff < self.md.x_incr)

    def on_floor(self):
        return (not self.md.can_fall())

    def policy_step(self):

        if ((self.start_cell == None) or (self.target_cell == None)):
            self.start_cell = self.md.get_player_cell()
            self.target_cell = self.get_target_cell(self.start_cell)

        if (self.close_enough_x(self.target_cell)):
            if (self.on_floor()):
                self.done = True
                self.start_cell = None
                self.target_cell = None
            return ACTION_NOP

        return ACTION_RIGHT


class interact_option(_Option):
    def __init__(self, md, drawer=None):
        _Option.__init__(self, md, drawer)

    def can_run(self):
        for obj in self.md.objects:
            if (obj.near_enough(self.md.playerx, self.md.playery + Y_SCALE / 2)):
                if (isinstance(obj, Handle)):
                    return True
                elif (isinstance(obj, Bolt)):
                    if (self.md.player_got_key()):
                        return True

        return False

    def policy_step(self):

        self.done = True
        return ACTION_INTERACT
