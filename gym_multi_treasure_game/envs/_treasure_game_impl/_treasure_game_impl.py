import random

import pygame
import numpy as np
from gym_multi_treasure_game.envs._treasure_game_impl import _treasure_game_drawer
from gym_multi_treasure_game.envs._treasure_game_impl._move_options import go_left_option, go_right_option, \
    up_ladder_option, \
    down_ladder_option, interact_option, down_left_option, down_right_option, jump_left_option, jump_right_option
from gym_multi_treasure_game.envs._treasure_game_impl._objects import Door, Handle, Bolt, Key, GoldCoin
from gym_multi_treasure_game.envs._treasure_game_impl._constants import X_SCALE, Y_SCALE, OPEN_SPACE, WALL, DOOR, \
    LADDER, \
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP, ACTION_INTERACT, AGENT_DOOR_CLOSED, AGENT_GOLD, \
    AGENT_BOLT_LOCKED, AGENT_DOOR_OPEN, AGENT_BOLT_UNLOCKED, AGENT_HANDLE_DOWN, AGENT_HANDLE_UP, AGENT_KEY, \
    AGENT_OPEN_SPACE, AGENT_WALL, AGENT_LADDER, AGENT_NORMALISE_CONSTANT

JUMP_REWARD = -5
STEP_REWARD = -1


class _TreasureGameImpl:

    def __init__(self, domain_file, object_file, interaction_file):

        self.domain_file = domain_file
        self.object_file = object_file
        self.interaction_file = interaction_file

        self.description = self.get_file_description(domain_file)
        self.map = self.build_map()
        self.objects = self.read_objects(object_file)
        self.extract_interactives(interaction_file)

        (self.playerx, self.playery) = self.player_initial_position()

        self.player_bag = []

        self.x_incr = X_SCALE // 10
        self.y_incr = Y_SCALE // 10
        self.jump_ticker = 0
        self.player_width = X_SCALE // 2
        self.player_height = Y_SCALE
        self.facing_right = True

        self.total_actions = 0

    def reset_game(self):

        self.description = self.get_file_description(self.domain_file)
        self.map = self.build_map()
        self.objects = self.read_objects(self.object_file)
        self.extract_interactives(self.interaction_file)

        (self.playerx, self.playery) = self.player_initial_position()

        self.player_bag = []

        self.x_incr = X_SCALE // 10
        self.y_incr = Y_SCALE // 10
        self.jump_ticker = 0
        self.player_width = X_SCALE // 2
        self.player_height = Y_SCALE
        self.facing_right = True

        self.total_actions = 0

    def extract_interactives(self, fname):
        self.doors = []
        self.handles = []
        self.bolts = []

        for obj in self.objects:
            if (isinstance(obj, Door)):
                self.doors.append(obj)
            elif (isinstance(obj, Handle)):
                self.handles.append(obj)
            elif (isinstance(obj, Bolt)):
                self.bolts.append(obj)

        f = open(fname, 'r')

        for line in f:
            if (line.strip()):
                (type1, index1, bool1, type2, index2, bool2) = line.split()
                bool1 = bool1 == 'True'
                bool2 = bool2 == 'True'
                index1 = int(index1)
                index2 = int(index2)

                first_list = None
                second_list = None

                if (type1 == 'handle'):
                    first_list = self.handles
                elif (type1 == 'bolt'):
                    first_list = self.bolts
                elif (type1 == 'door'):
                    first_list = self.doors

                if (type2 == 'door'):
                    second_list = self.doors
                elif (type2 == 'handle'):
                    second_list = self.handles
                elif (type2 == 'bolt'):
                    second_list = self.bolts

                first_list[index1].set_trigger(bool1, second_list[index2], bool2)

        f.close()

    def read_objects(self, object_file):
        f = open(object_file, 'r')

        o_list = []

        tcx = self.cell_width
        tcy = self.cell_height

        for line in f:
            if (line.startswith('door')):
                words = line.split()
                cx = int(words[1])
                cy = int(words[2])
                closed = False
                if (words[3] == 'True'):
                    closed = True
                d = Door(cx, cy, tcx, tcy, self.map, closed)
                o_list.append(d)
            elif (line.startswith("key")):
                words = line.split()
                cx = int(words[1])
                cy = int(words[2])
                k = Key(cx, cy, tcx, tcy)
                o_list.append(k)
            elif (line.startswith("bolt")):
                words = line.split()
                cx = int(words[1])
                cy = int(words[2])
                locked = (words[3] == 'True')
                b = Bolt(cx, cy, tcx, tcy, locked)
                o_list.append(b)
            elif (line.startswith("gold")):
                words = line.split()
                cx = int(words[1])
                cy = int(words[2])
                g = GoldCoin(cx, cy, tcx, tcy)
                o_list.append(g)
            elif (line.startswith("handle")):
                words = line.split()
                cx = int(words[1])
                cy = int(words[2])
                up = (words[3] == 'True')
                left_facing = (cx > self.cell_width / 2)
                h = Handle(cx, cy, tcx, tcy, up, left_facing)
                o_list.append(h)

        f.close()
        return o_list

    def player_initial_position(self):

        noise_offset_x = int(random.gauss(0, X_SCALE / 24))
        noise_offset_y = int(abs(random.gauss(0, Y_SCALE / 36)))

        for y in range(0, self.cell_height):
            for x in range(0, self.cell_width):
                if (self.description[y][x] != WALL):
                    return ((x * X_SCALE) + (X_SCALE // 2) + noise_offset_x, (y * Y_SCALE) + noise_offset_y)

        return (0, 0)

    def get_file_description(self, fname):
        f = open(fname, 'r')

        lines = f.readlines()
        desc = []

        for line in lines:
            cells = []
            line = str.strip(line)
            for c in line:
                cells.append(c)

            desc.append(cells)

        f.close()

        self.cell_width = len(desc[0])
        self.cell_height = len(desc)

        self.width = self.cell_width * X_SCALE
        self.height = self.cell_height * Y_SCALE

        return desc

    def build_map(self):

        retmap = []

        for y in range(0, self.cell_height):
            row_so_far = []
            for x in range(0, self.cell_width):
                pos_char = [self.description[y][x]] * X_SCALE
                row_so_far = row_so_far + pos_char
            for yz in range(0, Y_SCALE):
                retmap.append(row_so_far)

        return retmap

    def object_type_at(self, x, y):

        if ((x >= self.width) or (x < 0)):
            return WALL
        if ((y >= self.height) or (y < 0)):
            return WALL

        return self.map[y][x]

    def object_type_at_cell(self, xc, yc):
        x = (xc * X_SCALE) + (X_SCALE // 2)
        y = (yc * Y_SCALE) + (Y_SCALE // 2)
        return self.object_type_at(x, y)

    def up_clear(self):
        for xoff in [-self.x_incr, 0, self.x_incr]:
            for yoff in range(-self.y_incr, 0):
                if (self.object_type_at(self.playerx + xoff, self.playery + yoff) != OPEN_SPACE):
                    return False

        return True

    def can_go_up(self):

        if self.playery <= 1:
            return False

        for yoff in [-self.y_incr, 0, Y_SCALE - self.y_incr]:
            for xoff in [-self.player_width // 2, self.player_width // 2]:
                if (self.object_type_at(self.playerx + xoff, self.playery + yoff) == LADDER):
                    return True

        return False

    def can_go_down(self):
        for yoff in np.arange(0, Y_SCALE + self.y_incr):
            for xoff in [-self.player_width // 2, self.player_width // 2]:
                if (self.object_type_at(self.playerx + xoff, self.playery + yoff) == LADDER):
                    return True
        return False

    def can_go_left(self):

        for yoff in [self.y_incr, Y_SCALE - self.y_incr]:
            if (self.object_type_at(self.playerx - (self.player_width // 2) - self.x_incr,
                                    self.playery + yoff) == WALL):
                return False
            if (self.object_type_at(self.playerx - (self.player_width // 2) - self.x_incr,
                                    self.playery + yoff) == DOOR):
                return False

        return True

    def can_go_right(self):

        for yoff in [self.y_incr, Y_SCALE - self.y_incr]:
            if (self.object_type_at(self.playerx + (self.player_width // 2) + self.x_incr,
                                    self.playery + yoff) == WALL):
                return False
            if (self.object_type_at(self.playerx + (self.player_width // 2) + self.x_incr,
                                    self.playery + yoff) == DOOR):
                return False

        return True

    def can_fall(self):
        for xoff in [-self.player_width // 2 + 2, -2 + self.player_width // 2]:
            for yoff in [0, Y_SCALE + 2]:
                if (self.object_type_at(self.playerx + xoff, self.playery + yoff) != OPEN_SPACE):
                    return False;
        return True

    def step(self, action):

        xdelta = 0
        ydelta = 0

        self.total_actions = self.total_actions + 1

        if (action == ACTION_UP):
            if (self.can_go_up()):
                ydelta = self.noisy(-self.y_incr)

        elif (action == ACTION_DOWN):
            if (self.can_go_down()):
                ydelta = self.noisy(self.y_incr)

        elif (action == ACTION_LEFT):
            if (self.can_go_left()):
                xdelta = self.noisy(-self.x_incr)
                self.facing_right = False

        elif (action == ACTION_RIGHT):
            if (self.can_go_right()):
                xdelta = self.noisy(self.x_incr)
                self.facing_right = True

        elif (action == ACTION_JUMP):
            if ((not self.can_go_down()) and (self.up_clear())):
                self.jump_ticker = 22
                if (random.random() > 0.25):
                    self.jump_ticker = 23

        elif (action == ACTION_INTERACT):
            for obj in self.objects:
                if (obj.near_enough(self.playerx, self.playery + Y_SCALE / 2)):
                    if (isinstance(obj, Handle)):
                        obj.flip()
                    elif (isinstance(obj, Bolt)):
                        if (self.player_got_key()):
                            self.try_unlock(obj)
                            self.drop_key()

        if (self.jump_ticker > 0):
            if (self.up_clear()):
                ydelta = -self.y_incr
            self.jump_ticker = self.jump_ticker - 1
        elif (self.can_fall()):
            self.jump_ticker = 0
            ydelta = self.y_incr

        self.playerx = self.playerx + xdelta

        if (self.can_fall() and (ydelta > 0)):
            while (ydelta > 0):
                self.playery = self.playery + 1
                ydelta = ydelta - 1
                if (not self.can_fall()):
                    ydelta = 0
        else:
            self.playery = self.playery + ydelta

        for obj in self.objects:
            if (isinstance(obj, Key) or isinstance(obj, GoldCoin)):
                if (obj.near_enough(self.playerx, self.playery + Y_SCALE / 2)):
                    obj.move_to(self.cell_width - 1 - len(self.player_bag), self.cell_height - 1)
                    self.player_bag.append(obj)

        if (action == ACTION_JUMP):
            return JUMP_REWARD
        else:
            return STEP_REWARD

    def noisy(self, val):
        between_val = val / 2.0
        if (val < between_val):
            return int(round(random.uniform(val, between_val)))
        else:
            return int(round(random.uniform(between_val, val)))

    def get_state(self):
        state_vec = []

        state_vec.append(float(self.playerx) / self.width)
        state_vec.append(float(self.playery) / self.height)

        for obj in self.objects:
            if (obj.has_state()):
                state_vec = state_vec + obj.get_state()

        return state_vec

    def get_state_descriptors(self):

        state_desc = []
        state_desc.append('playerx')
        state_desc.append('playery')
        handle_no = 1

        for obj in self.objects:
            if (obj.has_state()):
                vec = obj.get_state_descriptors()
                tname = type(obj).__name__

                # There are two handles, so number them.
                if isinstance(obj, Handle):
                    tname = tname + str(handle_no)
                    handle_no = handle_no + 1

                for v in vec:
                    state_desc.append(tname + '.' + v)

        return state_desc

    def is_object_at(self, xc, yc):
        for obj in self.objects:
            if obj.cx == xc and obj.cy == yc:
                if isinstance(obj, Handle) or (isinstance(obj, Door) and obj.door_closed()) or \
                        isinstance(obj, Bolt) or isinstance(obj, GoldCoin) or isinstance(obj, Key):
                    return True
        return False

    def is_closed_door_at(self, xc, yc):
        for obj in self.objects:
            if obj.cx == xc and obj.cy == yc:
                if isinstance(obj, Door) and obj.door_closed():
                    return True
        return False

    def player_got_key(self):
        for obj in self.player_bag:
            if (isinstance(obj, Key)):
                return True
        return False

    def player_got_goldcoin(self):
        for obj in self.player_bag:
            if (isinstance(obj, GoldCoin)):
                return True
        return False

    def try_unlock(self, obj):
        # if(random.random() > 0.3):
        obj.unlock()

    def drop_key(self):
        for obj in self.player_bag:
            if (isinstance(obj, Key)):
                self.player_bag.remove(obj)
                obj.move_to(-1, -1)
                return

    def get_player_cell(self):
        xc = int(self.playerx // X_SCALE)
        yc = int((self.playery + (Y_SCALE // 2)) // Y_SCALE)

        return (xc, yc)

    def init_with_state(self, state):
        desc = self.get_state_descriptors()
        self.facing_right = True
        # Fill in the blanks
        st = self.get_state()
        for v in range(0, len(st)):
            if (state[v] == -99):
                state[v] = st[v]

        self.playerx = int(state[desc.index("playerx")] * self.width)
        self.playery = int(state[desc.index("playery")] * self.height)

        handle_no = 1

        for obj in self.objects:
            if (obj.has_state()):
                if (isinstance(obj, GoldCoin) or isinstance(obj, Key)):
                    x_label = type(obj).__name__ + ".x"
                    y_label = type(obj).__name__ + ".y"
                    xval = int(state[desc.index(x_label)] * self.width)
                    yval = int(state[desc.index(y_label)] * self.height)
                    obj.move_to_xy(xval, yval)
                elif (isinstance(obj, Handle)):
                    label = 'Handle' + str(handle_no) + '.angle'
                    angle = state[desc.index(label)]
                    obj.set_angle(angle)
                    obj.previously_triggered = True
                    handle_no = handle_no + 1
                elif (isinstance(obj, Bolt)):
                    val = (state[desc.index("Bolt.locked")] > 0.5)
                    obj.set_val(val)

        for obj in self.objects:
            if (isinstance(obj, Handle)):
                self.previously_triggered = False

    def _get_object(self, cell_x, cell_y):
        for o in self.objects:
            if o.cx == cell_x and o.cy == cell_y:
                if isinstance(o, Door):
                    return AGENT_DOOR_CLOSED if o.closed else AGENT_DOOR_OPEN
                if isinstance(o, GoldCoin):
                    if self.object_type_at_cell(cell_x, cell_y) != WALL:
                        return AGENT_GOLD
                    break
                if isinstance(o, Bolt):
                    return AGENT_BOLT_LOCKED if o.locked else AGENT_BOLT_UNLOCKED
                if isinstance(o, Key):
                    if self.object_type_at_cell(cell_x, cell_y) != WALL:
                        return AGENT_KEY
                    break
                if isinstance(o, Handle):
                    return AGENT_HANDLE_UP if o.is_up() else AGENT_HANDLE_DOWN
                else:
                    raise ValueError

        type = self.object_type_at_cell(cell_x, cell_y)
        if type == OPEN_SPACE:
            return AGENT_OPEN_SPACE
        elif type == WALL:
            return AGENT_WALL
        elif type == LADDER:
            return AGENT_LADDER
        else:
            raise ValueError

    def current_observation(self):
        state_vec = []
        (cell_x, cell_y) = self.get_player_cell()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                state_vec.append(self._get_object(cell_x + dx, cell_y + dy) / AGENT_NORMALISE_CONSTANT)  # normalise

        state_vec.extend([int(self.player_got_key()), int(self.player_got_goldcoin())])
        return np.array(state_vec)


def create_options(md, drawer=None):
    go_left = go_left_option(md, drawer)
    go_right = go_right_option(md, drawer)
    up_ladder = up_ladder_option(md, drawer)
    down_ladder = down_ladder_option(md, drawer)
    interact = interact_option(md, drawer)
    down_left = down_left_option(md, drawer)
    down_right = down_right_option(md, drawer)
    jump_left = jump_left_option(md, drawer)
    jump_right = jump_right_option(md, drawer)

    option_list = [go_left, go_right, up_ladder, down_ladder, interact, down_left, down_right, jump_left, jump_right]
    option_names = [o.__class__.__name__ for o in option_list]

    return (option_list, option_names)


if __name__ == "__main__":

    pygame.init()
    env = _TreasureGameImpl('domain.txt', 'domain-objects.txt', 'domain-interactions.txt')
    drawer = _treasure_game_drawer._TreasureGameDrawer(env, display_screen=True)

    clock = pygame.time.Clock()
    pygame.key.set_repeat()

    x = env.playerx
    y = env.playery

    saved_state = env.get_state()

    options, names = create_options(env, drawer=drawer)

    while not (env.player_got_goldcoin() and env.get_player_cell()[1] == 0):
        clock.tick(30)
        r = random.choice([x for x in options if x.can_run()])
        print(r.__class__.__name__)
        r.run()
        drawer.draw_domain()
        pygame.event.clear()
    pygame.display.quit()
