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
    #    0          1         2            3          4         5          6          7             8
    #  [go_left, go_right, up_ladder, down_ladder, interact, down_left, down_right, jump_left, jump_right]

    plan = [1, 3, 1, 6, 0, 3, 0, 1]

    v = run(env3, plan[:-1], plan[-1:])

    t = env3.render(mode='rgb_array')

    Image.save(make_image(v), 'test.png', mode='RGB')
    Image.save(t, 'test2.png', mode='RGB')

    x = 0