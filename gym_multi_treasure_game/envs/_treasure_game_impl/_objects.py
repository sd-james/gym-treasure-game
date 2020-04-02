# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:16:00 2014

@author: gdk
"""

import math
import random

from gym_multi_treasure_game.envs._treasure_game_impl._constants import X_SCALE, Y_SCALE

OPEN_SPACE = ' '
DOOR = 'D'

class _GameObject:

    def __init__(self, cx, cy, tcx, tcy):
        self.cx = cx
        self.cy = cy
        self.x = cx * X_SCALE
        self.y = cy * Y_SCALE
        self.radius = X_SCALE / 2

        self.world_height = tcy * Y_SCALE
        self.world_width = tcx * X_SCALE
        
        self.trigger_true = []
        self.trigger_true_vals = []
        self.trigger_false = []
        self.trigger_false_vals = []
        self.previously_triggered = False
                
    def move_to(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.x = cx * X_SCALE
        self.y = cy * Y_SCALE
        
    def move_to_xy(self, x, y):
        self.x = x
        self.y = y
        self.cx = int(x / X_SCALE)
        self.cy = int(y / Y_SCALE)
        
    def near_enough(self, x, y):
        
        centerx = self.x + (X_SCALE / 2)
        centery = self.y + (Y_SCALE / 2)
        dist = math.pow(x - centerx, 2) + math.pow(y - centery, 2)
        if(math.sqrt(dist) < self.radius):
            return True
        return False
        
#    def near_enough_prqint(self, x, y):
#        
#        centerx = self.x + (xscale / 2)
#        centery = self.y + (yscale / 2)
#        dist = math.pow(x - centerx, 2) + math.pow(y - centery, 2)
#        
#        print(str(centerx) + ", " + str(centery))
#        print(str(x) + ", " + str(y))
#        print(str(math.sqrt(dist))  + " : " + str(self.radius))
        
    def set_trigger(self, trig_val, obj, set_val):
        if(trig_val):
            self.trigger_true.append(obj)
            self.trigger_true_vals.append(set_val)
        else:
            self.trigger_false.append(obj)
            self.trigger_false_vals.append(set_val)
            
    def set_val(self, val):
        pass
            
    def process_trigger(self, val):
        self.previously_triggered = True

        if(val):
            for index in range(0, len(self.trigger_true)):
                obj = self.trigger_true[index]
                val_set = self.trigger_true_vals[index]
                
                if(not obj.previously_triggered):
                    obj.set_val(val_set)
        else:
            for index in range(0, len(self.trigger_false)):
                obj = self.trigger_false[index]
                val_set = self.trigger_false_vals[index]
                
                if(not obj.previously_triggered):
                    obj.set_val(val_set)
                
        self.previously_triggered = False
        
    def has_state(self):
        return False
        
    def get_state(self):
        return []
        
    def get_state_descriptors(self):
        return []

class Handle(_GameObject):
    def __init__(self, cx, cy, tcx, tcy, up = True, left_facing = False):
        _GameObject.__init__(self, cx, cy, tcx, tcy)
        self.up = up
        self.left_facing = left_facing
        
        if(self.up):
            self.angle = random.uniform(0.85, 1.0)
        else:
            self.angle = random.uniform(0, 0.15)
        self.radius = X_SCALE * 0.75
            
    def flip(self):
        
        if(random.uniform(0, 1) <= 0.8):
            self.set_val(not self.up)
            return True
        else:
            self.set_angle_wiggle()
            return False

    def get_angle(self):
        return self.angle
            
    def set_angle_wiggle(self):
        if(self.up):
            self.angle = random.uniform(0.85, 1.0)
        else:
            self.angle = random.uniform(0, 0.15)
            
    def set_angle(self, angle, propagate = True):
        old_up = self.up
        self.angle = angle
        
        if(self.angle <= 0.15):
            self.up = False
        else:
            self.up = True
    
        if((self.up != old_up) and propagate):
            self.process_trigger(self.up)            
            
    def set_val(self, val):
        if(self.up != val):
            self.up = val
            self.set_angle_wiggle()
            self.process_trigger(val)            
            
    def is_up(self):
        return self.up
        
    def has_state(self):
        return True
        
    def get_state(self):
        return [self.angle]
        
    def get_state_descriptors(self):
        return ['angle']
        

class Bolt(_GameObject):
    def __init__(self, cx, cy, tcx, tcy, locked=False):
        _GameObject.__init__(self, cx, cy, tcx, tcy)
        self.locked = locked
        
    def lock(self):
        self.set_val(True)
        
    def unlock(self):
        self.set_val(False)
        
    def set_val(self, val):
        if(self.locked != val):
            self.locked = val
            self.process_trigger(val)
        
    def get_locked(self):
        return self.locked
        
    def has_state(self):
        return True
        
    def get_state(self):
        if(self.locked):
            return [1.0]
        else:
            return [0.0]
        
    def get_state_descriptors(self):
        return ['locked']
        
class GoldCoin(_GameObject):
    def __init__(self, cx, cy, tcx, tcy):
        _GameObject.__init__(self, cx, cy, tcx, tcy)
        
    def has_state(self):
        return True
        
    def get_state(self):
        return [float(self.x) / self.world_width, float(self.y) / self.world_height]
        
    def get_state_descriptors(self):
        return ['x', 'y']

class Key(_GameObject):
    def __init__(self, cx, cy, tcx, tcy):
        _GameObject.__init__(self, cx, cy, tcx, tcy)
        
    def has_state(self):
        return True
        
    def get_state(self):
        return [float(self.x)/self.world_width, float(self.y)/self.world_height]
        
    def get_state_descriptors(self):
        return ['x', 'y']

class Door(_GameObject):
    def __init__(self, cx, cy, tcx, tcy, game_map, closed = True):
        _GameObject.__init__(self, cx, cy, tcx, tcy)
        self.game_map = game_map
        self.closed = closed
        self.update_map()
        
    def set_closed(self, closed):
        self.set_val(closed)
        
    def set_val(self, val):
        if(self.closed != val):
            self.closed = val
            self.update_map()
            self.process_trigger(val)
        
    def open_door(self):
        self.set_closed(False)

    def close_door(self):
        self.set_closed(True)
        
    def door_closed(self):
        return self.closed
        
    def update_map(self):
        character = OPEN_SPACE
        if(self.closed):
            character = DOOR
            
        row_chars = [character] * X_SCALE
        for y in range(0, Y_SCALE):
            self.game_map[self.cy * Y_SCALE + y][self.cx * X_SCALE:(self.cx + 1) * X_SCALE] = row_chars
            
    
