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


def run(env):
    env.reset()
    return env.views

if __name__ == '__main__':

    make_dir('gifs', clean=False)

    for i in range(1, 11):
        env3 = RecordableMultiTreasureGame(i, global_only=True)
        views = run(env3)
        Image.save(views[-1], 'gifs/Level-{}.png'.format(i), mode='RGB')
