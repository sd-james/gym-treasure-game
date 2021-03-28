
import itertools
import warnings
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gym_multi_treasure_game.envs.pca.base_pca import PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.utils import plan_parallel
from pyddl.pddl.domain import Domain
from pyddl.pddl.operator import Operator
from pyddl.pddl.predicate import Predicate
from s2s.hierarchy.discover_hddl_methods import _generate_mdp, _generate_full_mdp
from s2s.hierarchy.network import Node, Path
from s2s.hierarchy.option_discovery import _get_path
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import save, run_parallel, load

import matplotlib


MAPPER = load("/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/transfer/MAPPER.pkl")

x = list()
y = list()
for key, (has_key, has_gold) in MAPPER.items():

    x.append(key)
    if has_key == 0 and has_gold == 0:
        y.append(0)
    elif has_gold == 1:
        y.append(2)
    else:
        y.append(1)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x, y)
save(neigh, '/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/transfer/knn.pkl')



