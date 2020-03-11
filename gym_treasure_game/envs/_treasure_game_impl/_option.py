# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:10:26 2014

@author: gdk
"""

class _Option:
    def __init__(self, md, drawer = None):
        self.md = md
        self.drawer = drawer
        self.done = False        
        
    def can_run(self):
        pass
    
    def policy_step(self):
        pass
    
    def run(self):

        if(not self.can_run()):
            return None

        self.done = False
        totrew = 0
        
        while(not self.done):
            act = self.policy_step()
            rew = self.md.step(act)
            totrew = totrew + rew
            
            if(self.drawer != None):
                self.drawer.draw_domain()

        return totrew