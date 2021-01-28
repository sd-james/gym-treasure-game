# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:10:26 2014

@author: gdk
"""
import warnings

from gym_multi_treasure_game.envs.treasure_game_impl_._constants import BANNER, TORCH, OPEN_SPACE


class _Option:

    def __init__(self, md, drawer = None):
        self.md = md
        self.drawer = drawer
        self.done = False        
        
    def can_run(self):
        pass
    
    def policy_step(self):
        pass

    def empty(self, x, y):
        type = self.md.object_type_at_cell(x, y)
        return type in [OPEN_SPACE, TORCH, BANNER]

    def run(self):

        if(not self.can_run()):
            return None

        self.done = False
        totrew = 0
        
        while(not self.done):
            act = self.policy_step()
            rew = self.md.step(act)
            totrew = totrew + rew

            if totrew < -1000:
                raise ValueError('We are stuck!!')

            if(self.drawer != None):
                self.drawer.draw_domain()

        return totrew