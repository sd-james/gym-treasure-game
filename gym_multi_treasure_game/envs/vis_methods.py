import random

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageChops
from pygifsicle import optimize

from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.recordable_multi_treasure_game import RecordableMultiTreasureGame
from s2s.image import Image
from s2s.utils import make_dir


def run(env, start_plan, end_plan, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    # get in position
    env.reset()
    for x in start_plan:
        env.step(x)
    # clear views
    env.reset_view()
    for x in end_plan:
        env.step(x)
    return env.views


#    0          1         2            3          4         5          6          7             8
#  [go_left, go_right, up_ladder, down_ladder, interact, down_left, down_right, jump_left, jump_right]


########################################### LEVEL 1 ##############################################################


def _a_1(env):
    """
    up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1
    (:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-0-5b83f3d4-58f8-4978-8dc4-8f91ecab30d9
        :parameters ()
        :task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
        :precondition (and (notfailed) (psymbol_0) (symbol_3))
        :ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
    )
    [2, 1, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3]
    new_plan = [2]
    return run(env, plan, new_plan)


def _a_2(env):
    """
    up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1
    (:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-0-5b83f3d4-58f8-4978-8dc4-8f91ecab30d9
        :parameters ()
        :task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
        :precondition (and (notfailed) (psymbol_0) (symbol_3))
        :ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
    )
    [2, 1, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 2]
    new_plan = [1]
    return run(env, plan, new_plan)


def _a_3(env):
    """
    up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1
    (:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-0-5b83f3d4-58f8-4978-8dc4-8f91ecab30d9
        :parameters ()
        :task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
        :precondition (and (notfailed) (psymbol_0) (symbol_3))
        :ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
    )
    [2, 1, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 2, 1]
    new_plan = [8]
    return run(env, plan, new_plan)


def _a_4(env):
    """
    up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1
    (:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-0-5b83f3d4-58f8-4978-8dc4-8f91ecab30d9
        :parameters ()
        :task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
        :precondition (and (notfailed) (psymbol_0) (symbol_3))
        :ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
    )
    [2, 1, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 2, 1, 8]
    new_plan = [8]
    return run(env, plan, new_plan)


def _b_1(env):
    """
	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-1-3afdcdd8-c490-4fa0-85fa-cb7531291a76
		:parameters ()
		:task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (symbol_1) (psymbol_2) (symbol_3))
		:ordered-tasks (and (go_right_option-6a5197e1-33af-4c0a-ab6b-1f3c6b9540f5) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
	)
	[1, 6, 8, 8]
    """
    plan = [1, 3]
    new_plan = [1]
    return run(env, plan, new_plan, seed=10)


def _b_2(env):
    """
	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-1-3afdcdd8-c490-4fa0-85fa-cb7531291a76
		:parameters ()
		:task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (symbol_1) (psymbol_2) (symbol_3))
		:ordered-tasks (and (go_right_option-6a5197e1-33af-4c0a-ab6b-1f3c6b9540f5) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
	)
	[1, 6, 8, 8]
    """
    plan = [1, 3, 1]
    new_plan = [6]
    return run(env, plan, new_plan, seed=10)


def _b_3(env):
    """
	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-1-3afdcdd8-c490-4fa0-85fa-cb7531291a76
		:parameters ()
		:task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (symbol_1) (psymbol_2) (symbol_3))
		:ordered-tasks (and (go_right_option-6a5197e1-33af-4c0a-ab6b-1f3c6b9540f5) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
	)
	[1, 6, 8, 8]
    """
    plan = [1, 3, 1, 6]
    new_plan = [8]
    return run(env, plan, new_plan, seed=10)


def _b_4(env):
    """
	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-1-3afdcdd8-c490-4fa0-85fa-cb7531291a76
		:parameters ()
		:task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (symbol_1) (psymbol_2) (symbol_3))
		:ordered-tasks (and (go_right_option-6a5197e1-33af-4c0a-ab6b-1f3c6b9540f5) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
	)
	[1, 6, 8, 8]
    """
    plan = [1, 3, 1, 6, 8]
    new_plan = [8]
    return run(env, plan, new_plan, seed=10)


def _c_1(env):
    """
    up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea
    	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1-0-1d38ea0e-ac4e-4685-9811-566925fcdb8c
    		:parameters ()
    		:task (up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1)
    		:precondition (and (notfailed) (psymbol_0) (symbol_3))
    		:ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_left_option-38bd5ea7-532b-4550-b6ed-d7fb217091b7))
    	)
    	[2, 1, 8, 7]
    """
    plan = [1, 3, 1, 6, 0, 3]
    new_plan = [2]
    return run(env, plan, new_plan)


def _c_2(env):
    """
    go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3
    	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1-0-1d38ea0e-ac4e-4685-9811-566925fcdb8c
    		:parameters ()
    		:task (up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1)
    		:precondition (and (notfailed) (psymbol_0) (symbol_3))
    		:ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_left_option-38bd5ea7-532b-4550-b6ed-d7fb217091b7))
    	)
    	[2, 1, 8, 7]
    """
    plan = [1, 3, 1, 6, 0, 3, 2]
    new_plan = [1]
    return run(env, plan, new_plan)


def _c_3(env):
    """
    jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580
    	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1-0-1d38ea0e-ac4e-4685-9811-566925fcdb8c
    		:parameters ()
    		:task (up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1)
    		:precondition (and (notfailed) (psymbol_0) (symbol_3))
    		:ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_left_option-38bd5ea7-532b-4550-b6ed-d7fb217091b7))
    	)
    	[2, 1, 8, 7]
    """
    plan = [1, 3, 1, 6, 0, 3, 2, 1]
    new_plan = [8]
    return run(env, plan, new_plan)


def _c_4(env):
    """
    jump_left_option-38bd5ea7-532b-4550-b6ed-d7fb217091b7
    	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1-0-1d38ea0e-ac4e-4685-9811-566925fcdb8c
    		:parameters ()
    		:task (up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1)
    		:precondition (and (notfailed) (psymbol_0) (symbol_3))
    		:ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_left_option-38bd5ea7-532b-4550-b6ed-d7fb217091b7))
    	)
    	[2, 1, 8, 7]
    """
    plan = [1, 3, 1, 6, 0, 3, 2, 1, 8]
    new_plan = [7]
    return run(env, plan, new_plan)


def _d_1(env):
    """
    go_right_option-df783eff-fc68-4a93-bb75-ee3d4e100c23
	(:method m-down_left_option-jump_right_option-Level-1-11-c2ee8da8-bab0-4582-a037-7920e0f4393c
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (symbol_31) (notfailed) (symbol_1) (psymbol_3))
		:ordered-tasks (and (go_right_option-df783eff-fc68-4a93-bb75-ee3d4e100c23) (go_right_option-cd72a1b5-0547-43dd-be33-592dac9974bb) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933))
	)
	[1, 1, 6, 8]
    """
    plan = [1, 3, 0]
    new_plan = [1]
    return run(env, plan, new_plan)


def _d_2(env):
    """
    go_right_option-cd72a1b5-0547-43dd-be33-592dac9974bb
	(:method m-down_left_option-jump_right_option-Level-1-11-c2ee8da8-bab0-4582-a037-7920e0f4393c
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (symbol_31) (notfailed) (symbol_1) (psymbol_3))
		:ordered-tasks (and (go_right_option-df783eff-fc68-4a93-bb75-ee3d4e100c23) (go_right_option-cd72a1b5-0547-43dd-be33-592dac9974bb) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933))
	)
	[1, 1, 6, 8]
    """
    plan = [1, 3, 0, 1]
    new_plan = [1]
    return run(env, plan, new_plan)


def _d_3(env):
    """
    down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97
	(:method m-down_left_option-jump_right_option-Level-1-11-c2ee8da8-bab0-4582-a037-7920e0f4393c
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (symbol_31) (notfailed) (symbol_1) (psymbol_3))
		:ordered-tasks (and (go_right_option-df783eff-fc68-4a93-bb75-ee3d4e100c23) (go_right_option-cd72a1b5-0547-43dd-be33-592dac9974bb) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933))
	)
	[1, 1, 6, 8]
    """
    plan = [1, 3, 0, 1, 1]
    new_plan = [6, 8]
    return run(env, plan, new_plan)


def _d_4(env):
    """
    jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933
	(:method m-down_left_option-jump_right_option-Level-1-11-c2ee8da8-bab0-4582-a037-7920e0f4393c
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (symbol_31) (notfailed) (symbol_1) (psymbol_3))
		:ordered-tasks (and (go_right_option-df783eff-fc68-4a93-bb75-ee3d4e100c23) (go_right_option-cd72a1b5-0547-43dd-be33-592dac9974bb) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933))
	)
	[1, 1, 6, 8]
    """
    plan = [1, 3, 0, 1, 1, 6]
    new_plan = [8]
    return run(env, plan, new_plan)


def _dd_1(env):
    """
    go_right_option-88dcce14-f5a4-4eca-9f6b-4f7973475c86
	(:method m-down_left_option-jump_right_option-Level-1-7-32882f1c-fd51-4199-826d-af884ff42b26
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (psymbol_19) (notfailed) (symbol_33))
		:ordered-tasks (and (go_right_option-88dcce14-f5a4-4eca-9f6b-4f7973475c86) (jump_right_option-9075f1c6-cec0-4509-990e-375f52911f7c))
	)
	[1, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 2]
    new_plan = [1]
    return run(env, plan, new_plan)


def _dd_2(env):
    """
    jump_right_option-9075f1c6-cec0-4509-990e-375f52911f7c
	(:method m-down_left_option-jump_right_option-Level-1-7-32882f1c-fd51-4199-826d-af884ff42b26
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (psymbol_19) (notfailed) (symbol_33))
		:ordered-tasks (and (go_right_option-88dcce14-f5a4-4eca-9f6b-4f7973475c86) (jump_right_option-9075f1c6-cec0-4509-990e-375f52911f7c))
	)
	[1, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 2, ]
    new_plan = [8]
    return run(env, plan, new_plan)


def _fff_1(env):
    """
    jump_right_option-2d5e3857-b6f0-466a-b6ca-2919f7389b86
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-1-c9d25672-115b-420f-ab34-be9410c22bca
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_41) (notfailed) (psymbol_15))
		:ordered-tasks (and (jump_right_option-2d5e3857-b6f0-466a-b6ca-2919f7389b86) (jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b) (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95))
	)
	[8, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 1]
    new_plan = [8]
    return run(env, plan, new_plan)


def _fff_2(env):
    """
    jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-1-c9d25672-115b-420f-ab34-be9410c22bca
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_41) (notfailed) (psymbol_15))
		:ordered-tasks (and (jump_right_option-2d5e3857-b6f0-466a-b6ca-2919f7389b86) (jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b) (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95))
	)
	[8, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 1, 8]
    new_plan = [8]
    return run(env, plan, new_plan)


def _fff_3(env):
    """
    jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-1-c9d25672-115b-420f-ab34-be9410c22bca
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_41) (notfailed) (psymbol_15))
		:ordered-tasks (and (jump_right_option-2d5e3857-b6f0-466a-b6ca-2919f7389b86) (jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b) (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95))
	)
	[8, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 1, 8, 8]
    new_plan = [8]
    return run(env, plan, new_plan)


def _ff_1(env):
    """
    go_right_option-959e2fcc-bbfa-4894-a389-abf5e970b4f4
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-0-2a0a3598-9767-4d5b-955d-f50842e239fd
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (psymbol_0) (symbol_3))
		:ordered-tasks (and (go_right_option-959e2fcc-bbfa-4894-a389-abf5e970b4f4) (jump_right_option-2d5e3857-b6f0-466a-b6ca-2919f7389b86) (jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b) (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95))
	)
	[1, 8, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1]
    new_plan = [1]
    return run(env, plan, new_plan)


def _f_2(env):
    """
    go_right_option-a81f92d4-cc9d-4f7c-9567-0496f00aa66a
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-2-7ee5fb65-3bf2-4cd4-898e-5ab4c46cc7c4
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_22) (symbol_36) (psymbol_40) (notfailed))
		:ordered-tasks (and (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95) (go_right_option-a81f92d4-cc9d-4f7c-9567-0496f00aa66a) (interact_option-edbd2400-3935-4b4e-a772-da8ba93f4b88) (go_left_option-8022db16-eb67-490d-a45a-221d0f37adb9))
	)
	[8, 1, 4, 0]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 1, 8, 8, 8]
    new_plan = [1]
    return run(env, plan, new_plan)


def _f_3(env):
    """
    interact_option
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-2-7ee5fb65-3bf2-4cd4-898e-5ab4c46cc7c4
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_22) (symbol_36) (psymbol_40) (notfailed))
		:ordered-tasks (and (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95) (go_right_option-a81f92d4-cc9d-4f7c-9567-0496f00aa66a) (interact_option-edbd2400-3935-4b4e-a772-da8ba93f4b88) (go_left_option-8022db16-eb67-490d-a45a-221d0f37adb9))
	)
	[8, 1, 4, 0]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 1, 8, 8, 8, 1]
    new_plan = [4, ]
    return run(env, plan, new_plan)


def _f_4(env):
    """
    go_left_option-8022db16-eb67-490d-a45a-221d0f37adb9
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-2-7ee5fb65-3bf2-4cd4-898e-5ab4c46cc7c4
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_22) (symbol_36) (psymbol_40) (notfailed))
		:ordered-tasks (and (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95) (go_right_option-a81f92d4-cc9d-4f7c-9567-0496f00aa66a) (interact_option-edbd2400-3935-4b4e-a772-da8ba93f4b88) (go_left_option-8022db16-eb67-490d-a45a-221d0f37adb9))
	)
	[8, 1, 4, 0]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 1, 8, 8, 8, 1, 4]
    new_plan = [0]
    return run(env, plan, new_plan)


########################################### LEVEL 2 ##############################################################

def a(env):
    """
    up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1
    (:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-0-5b83f3d4-58f8-4978-8dc4-8f91ecab30d9
        :parameters ()
        :task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
        :precondition (and (notfailed) (psymbol_0) (symbol_3))
        :ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
    )
    [2, 1, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3]
    new_plan = [2, 1, 8, 8]
    return run(env, plan, new_plan)


def b(env):
    """
	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1-1-3afdcdd8-c490-4fa0-85fa-cb7531291a76
		:parameters ()
		:task (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (symbol_1) (psymbol_2) (symbol_3))
		:ordered-tasks (and (go_right_option-6a5197e1-33af-4c0a-ab6b-1f3c6b9540f5) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933) (jump_right_option-6f369c7e-7a5b-43d1-b931-7a7e1a7efd48))
	)
	[1, 6, 8, 8]
    """
    plan = [1, 3]
    new_plan = [1, 6, 8, 8]
    return run(env, plan, new_plan, seed=10)


def c(env):
    """
    	(:method m-up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1-0-1d38ea0e-ac4e-4685-9811-566925fcdb8c
    		:parameters ()
    		:task (up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1)
    		:precondition (and (notfailed) (psymbol_0) (symbol_3))
    		:ordered-tasks (and (up_ladder_option-1b47b881-c5ee-4606-b54b-a3e0988825ea) (go_right_option-654976ff-8cbc-442e-a906-16e217b7e2a3) (jump_right_option-1f98b3c3-aa53-43a8-8fc7-a5bb16c85580) (jump_left_option-38bd5ea7-532b-4550-b6ed-d7fb217091b7))
    	)
    	[2, 1, 8, 7]
    """
    plan = [1, 3, 1, 6, 0, 3]
    new_plan = [2, 1, 8, 7]
    return run(env, plan, new_plan)


def d(env):
    """
	(:method m-down_left_option-jump_right_option-Level-1-11-c2ee8da8-bab0-4582-a037-7920e0f4393c
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (symbol_31) (notfailed) (symbol_1) (psymbol_3))
		:ordered-tasks (and (go_right_option-df783eff-fc68-4a93-bb75-ee3d4e100c23) (go_right_option-cd72a1b5-0547-43dd-be33-592dac9974bb) (down_right_option-2fe9c658-b6c8-4f64-9b57-8475dd619b97) (jump_right_option-164b12a8-bd71-4c3b-906b-4eff49391933))
	)
	[1, 1, 6, 8]
    """
    plan = [1, 3, 0]
    new_plan = [1, 1, 6, 8]
    return run(env, plan, new_plan)


def dd(env):
    """
	(:method m-down_left_option-jump_right_option-Level-1-7-32882f1c-fd51-4199-826d-af884ff42b26
		:parameters ()
		:task (down_left_option-jump_right_option-Level-1)
		:precondition (and (psymbol_19) (notfailed) (symbol_33))
		:ordered-tasks (and (go_right_option-88dcce14-f5a4-4eca-9f6b-4f7973475c86) (jump_right_option-9075f1c6-cec0-4509-990e-375f52911f7c))
	)
	[1, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 2]
    new_plan = [1, 8]
    return run(env, plan, new_plan)


def e(env):
    """
go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1

	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-0-94c5a531-7228-4341-bd57-d15eb2905599
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (psymbol_18) (symbol_4))
		:ordered-tasks (and (go_right_option-7050ae47-661c-4a54-b462-ae0b2224170d) (jump_right_option-eae6d588-35c6-4c22-bbfc-64b91dbd58b2) (jump_right_option-2652f0dd-521f-479c-9c44-141dce2ed7cc) (jump_right_option-e7c5c5a1-8402-415f-8073-e521c996718d))
	)
	[1, 8, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0]
    new_plan = [1, 8, 8, 8]
    return run(env, plan, new_plan)


def f(env):
    """
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-2-7ee5fb65-3bf2-4cd4-898e-5ab4c46cc7c4
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_22) (symbol_36) (psymbol_40) (notfailed))
		:ordered-tasks (and (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95) (go_right_option-a81f92d4-cc9d-4f7c-9567-0496f00aa66a) (interact_option-edbd2400-3935-4b4e-a772-da8ba93f4b88) (go_left_option-8022db16-eb67-490d-a45a-221d0f37adb9))
	)
	[8, 1, 4, 0]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 1, 8, 8]
    new_plan = [8, 1, 4, 0]
    return run(env, plan, new_plan)


def ff(env):
    """
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-0-2a0a3598-9767-4d5b-955d-f50842e239fd
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (notfailed) (psymbol_0) (symbol_3))
		:ordered-tasks (and (go_right_option-959e2fcc-bbfa-4894-a389-abf5e970b4f4) (jump_right_option-2d5e3857-b6f0-466a-b6ca-2919f7389b86) (jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b) (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95))
	)
	[1, 8, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1]
    new_plan = [1, 8, 8, 8]
    return run(env, plan, new_plan)


def fff(env):
    """
	(:method m-go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1-1-c9d25672-115b-420f-ab34-be9410c22bca
		:parameters ()
		:task (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1)
		:precondition (and (symbol_41) (notfailed) (psymbol_15))
		:ordered-tasks (and (jump_right_option-2d5e3857-b6f0-466a-b6ca-2919f7389b86) (jump_right_option-d1096fa0-f66d-4dd5-8ff1-56484ffdef7b) (jump_right_option-5916ac9a-d013-44f5-8faf-006cd2df4a95))
	)
	[8, 8, 8]
    """
    plan = [1, 3, 1, 6, 0, 3, 1]
    new_plan = [8, 8, 8]
    return run(env, plan, new_plan)


########################################### LEVEL 3 ##############################################################


def g(env):
    """
	(:method m-down_left_option-down_left_option-Level-2-2-57bbec5f-a447-48f2-a53f-07d7399aa949
		:parameters ()
		:task (down_left_option-down_left_option-Level-2)
		:precondition (and (symbol_24) (symbol_1) (notfailed) (psymbol_13))
		:ordered-tasks (and (down_right_option-jump_left_option-Level-1) (down_left_option-go_left_option-down_ladder_option-Level-1))
	)
	[3, 0, 5, 7, 5, 0, 3]
    """
    plan = [1, 1]
    new_plan = [3, 0, 5, 7, 5, 0, 3]
    return run(env, plan, new_plan)


def h(env):
    """
	(:method m-down_left_option-jump_right_option-Level-2-0-79b70a44-8e8a-4407-81a4-f08afd448090
		:parameters ()
		:task (down_left_option-jump_right_option-Level-2)
		:precondition (and (symbol_1) (psymbol_14) (symbol_28) (notfailed))
		:ordered-tasks (and (down_left_option-go_left_option-down_ladder_option-Level-1) (jump_right_option-go_right_option-up_ladder_option-Level-1))
	)
	[6, 0, 3, 0, 1, 2]
    """
    plan = [1, 3, 1]
    new_plan = [6, 0, 3, 0, 1, 2]
    return run(env, plan, new_plan)


def i(env):
    """
    (:method m-down_left_option-jump_right_option-Level-2-3-6217e843-dd3e-434c-bf33-c6a864bd3b39
     :parameters ()
    : task(down_left_option - jump_right_option - Level - 2)
    :precondition( and (symbol_26)(symbol_1)(notfailed)(psymbol_29))
    :ordered - tasks( and (down_left_option - go_left_option - Level - 1)(
        jump_right_option - go_right_option - up_ladder_option - Level - 1))
    )
    [5, 7, 5, 0, 3, 0, 1, 2]
    """
    plan = [1, 1, 3, 0]
    new_plan = [5, 7, 5, 0, 3, 0, 1, 2]
    return run(env, plan, new_plan)


def j(env):
    """
	(:method m-down_left_option-up_ladder_option-Level-2-0-c69f8a5c-bbdd-4737-b62b-4353db89dba2
		:parameters ()
		:task (down_left_option-up_ladder_option-Level-2)
		:precondition (and (symbol_24) (symbol_1) (notfailed) (psymbol_13))
		:ordered-tasks (and (down_left_option-go_left_option-down_ladder_option-Level-1) (up_ladder_option-go_right_option-jump_right_option-jump_right_option-Level-1))
	)
	[0, 3, 1, 6, 8, 8]
    """
    plan = [1, 1]
    new_plan = [0, 3, 1, 6, 8, 8]
    return run(env, plan, new_plan)


def k(env):
    """
	(:method m-go_right_option-go_right_option-go_right_option-up_ladder_option-Level-2-0-264e8685-93df-4302-95fa-00c80a96b480
		:parameters ()
		:task (go_right_option-go_right_option-go_right_option-up_ladder_option-Level-2)
		:precondition (and (symbol_24) (notfailed) (psymbol_8) (symbol_22))
		:ordered-tasks (and (go_right_option-jump_right_option-jump_right_option-Level-1) (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1) (go_right_option-go_left_option-Level-1) (up_ladder_option-go_right_option-jump_right_option-jump_left_option-Level-1))
	)
	[3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8, 7]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 2]
    new_plan = [3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8, 7]
    return run(env, plan, new_plan)


def l(env):
    """
	(:method m-down_left_option-jump_right_option-Level-2-0-5e47d170-dafe-4a45-94ff-95f708f7e248
		:parameters ()
		:task (down_left_option-jump_right_option-Level-2)
		:precondition (and (psymbol_14) (symbol_28) (notfailed))
		:ordered-tasks (and (down_left_option-jump_right_option-Level-1) (jump_right_option-go_right_option-up_ladder_option-Level-1))
	)
	[6, 8, 8, 1, 2]
    """
    plan = [1, 3, 1]
    new_plan = [6, 8, 8, 1, 2]
    return run(env, plan, new_plan, seed=3)


def m(env):
    """
	(:method m-down_left_option-jump_right_option-Level-2-1-cd4eebca-ada6-42b1-9063-85f550a80d7c
		:parameters ()
		:task (down_left_option-jump_right_option-Level-2)
		:precondition (and (symbol_3) (notfailed) (psymbol_0))
		:ordered-tasks (and (down_left_option-jump_right_option-Level-1) (jump_right_option-go_right_option-up_ladder_option-Level-1))
	)
	[2, 1, 8, 8, 1, 2]
    """
    plan = [1, 3, 1, 6, 0, 3]
    new_plan = [2, 1, 8, 8, 1, 2]
    return run(env, plan, new_plan)


def n(env):
    """
	(:method m-down_left_option-jump_left_option-Level-2-2-921fbbd7-39e0-4aa3-8c6a-2798baded2b1
		:parameters ()
		:task (down_left_option-jump_left_option-Level-2)
		:precondition (and (symbol_24) (notfailed) (psymbol_8) (symbol_22))
		:ordered-tasks (and (go_right_option-jump_right_option-jump_right_option-Level-1) (go_right_option-jump_right_option-jump_right_option-jump_right_option-Level-1) (go_right_option-jump_right_option-jump_right_option-down_left_option-Level-1) (jump_left_option-go_left_option-down_ladder_option-Level-1))
	)
	[3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 3]
    """
    plan = [1, 3, 1, 6, 0, 3, 0, 1, 2]
    new_plan = [3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 3]
    return run(env, plan, new_plan)


########################################### LEVEL 4 ##############################################################

def w(env):
    """
	(:method m-down_left_option-down_left_option-Level-3-0-ffa24df6-4486-430e-ab61-5a0c3efdf2e5
		:parameters ()
		:task (down_left_option-down_left_option-Level-3)
		:precondition (and (symbol_26) (notfailed) (psymbol_29) (symbol_1))
		:ordered-tasks (and (down_left_option-jump_left_option-Level-2) (down_left_option-jump_right_option-Level-2))
	)
	[5, 7, 5, 8, 7, 0, 1, 6, 8, 8, 1, 2]
    """
    plan = [1, 1, 3, 0]
    new_plan = [5, 7, 5, 8, 7, 0, 1, 6, 8, 8, 1, 2]
    return run(env, plan, new_plan, seed=10)


def x(env):
    """
	(:method m-down_left_option-go_right_option-Level-3-0-7ac981b1-483f-466c-bfad-39d17452d5ff
		:parameters ()
		:task (down_left_option-go_right_option-Level-3)
		:precondition (and (notfailed) (psymbol_13) (symbol_24) (symbol_1))
		:ordered-tasks (and (down_left_option-jump_right_option-Level-2) (go_right_option-go_right_option-go_right_option-down_left_option-Level-2))
	)
	[3, 0, 5, 7, 5, 0, 3, 0, 1, 2, 3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8]
    """
    plan = [1, 3, 1, 6, 0, 3]
    new_plan = [3, 0, 5, 7, 5, 0, 3, 0, 1, 2, 3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8]
    return run(env, plan, new_plan)


def y(env):
    """
	(:method m-down_left_option-go_right_option-Level-3-1-2eab8e2c-24d6-4011-b419-df70540dfdf4
		:parameters ()
		:task (down_left_option-go_right_option-Level-3)
		:precondition (and (symbol_26) (notfailed) (psymbol_29) (symbol_1))
		:ordered-tasks (and (down_left_option-jump_right_option-Level-2) (go_right_option-go_right_option-go_right_option-down_left_option-Level-2))
	)
	[5, 7, 5, 0, 3, 0, 1, 2, 3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8]
    """
    plan = [1, 1, 3, 0]
    new_plan = [3, 0, 5, 7, 5, 0, 3, 0, 1, 2, 3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8]
    return run(env, plan, new_plan)


def z(env):
    """
	(:method m-down_left_option-go_right_option-Level-3-0-7ac981b1-483f-466c-bfad-39d17452d5ff
		:parameters ()
		:task (down_left_option-go_right_option-Level-3)
		:precondition (and (notfailed) (psymbol_13) (symbol_24) (symbol_1))
		:ordered-tasks (and (down_left_option-jump_right_option-Level-2) (go_right_option-go_right_option-go_right_option-down_left_option-Level-2))
	)
	[3, 0, 5, 7, 5, 0, 3, 0, 1, 2, 3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8]
    """
    plan = [1, 1]
    new_plan = [3, 0, 5, 7, 5, 0, 3, 0, 1, 2, 3, 1, 8, 8, 8, 1, 4, 0, 5, 5, 5, 0, 2, 1, 8]
    return run(env, plan, new_plan)


def make_image(views):
    return views[-1]


def make_video(file: str, frames):
    height, width, layers = np.array(frames[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(file, fourcc, 60, (width, height))
    for frame in frames:
        # writer.writeFrame(frame[:, :, ::-1])
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
    # writer.close()  # close the writer


if __name__ == '__main__':

    make_dir('gifs', clean=False)

    pca = PCA(PCA_STATE)
    pca.load('pca/models/dropped_key_pca_state.dat')
    pca2 = PCA(PCA_INVENTORY)
    pca2.load('pca/models/dropped_key_pca_inventory.dat')
    pcas = [pca, pca2]
    env = RecordableMultiTreasureGame(1, pcas=pcas)
    env2 = RecordableMultiTreasureGame(1)
    env3 = RecordableMultiTreasureGame(1, global_only=True)
    env4 = RecordableMultiTreasureGame(1, global_only=True, alpha=True, redraw=False)

    functions = []
    for key, value in list(locals().items()):
        if callable(value) and value.__module__ == __name__:
            if key[0] == '_':
                # if len(key) <= 3 and key != 'run':
                functions.append(value)

    for func in functions:
        name = func.__name__
        views = func(env3)
        make_video('gifs/{}.mp4'.format(name), views)
        # imageio.mimsave('gifs/{}-global.gif'.format(func.__name__), views, fps=60)
        # optimize('gifs/{}-global.gif'.format(func.__name__))
        views = func(env4)
        make_video('gifs/{}-trajectory.mp4'.format(name), views)
        Image.save(make_image(views), 'gifs/{}.png'.format(name), mode='RGB')
