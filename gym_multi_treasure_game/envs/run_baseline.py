import warnings
from functools import partial
from typing import List

import cv2
import imageio
import numpy as np
from tqdm import trange

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.evaluate import evaluate_model
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.clause import Clause
from pyddl.pddl.domain import Domain
from pyddl.pddl.predicate import Predicate
from s2s import utils
from s2s.core.build_model import build_model
from s2s.core.link_operators import link_pddl
from s2s.env.s2s_env import View
from s2s.env.walkthrough_env import FourFourRoomsEnv
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.planner.mgpt_planner import mGPT, PlanOutput
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import make_dir, make_path, save, load, run_parallel

if __name__ == '__main__':

    TASK = 1
    pca = PCA(PCA_STATE)

    pca.load('pca/models/mod_no_fancy_pca_state.dat')

    pca2 = PCA(PCA_INVENTORY)

    pca2.load('pca/models/mod_no_fancy_pca_inventory.dat')

    env = PCAWrapper(MultiTreasureGame(TASK, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
    save_dir = '../data.bak/{}_gen_positive'.format(TASK)
    make_dir(save_dir, clean=False)

    vals = list()
    for _ in range(10):
        n_samples = np.inf
        for n_eps in trange(20, 50):
            try:
                domain, problem, info = build_model(env,
                                                    load_existing_data='/media/hdd/treasure_data/{}'.format(TASK),
                                                    save_dir=None,
                                                    n_jobs=16,
                                                    seed=0,
                                                    n_episodes=n_eps,
                                                    options_per_episode=1000,
                                                    view=View.AGENT,
                                                    **CONFIG[TASK],
                                                    visualise=False,
                                                    verbose=False)
                if evaluate_model(env, domain, problem) == 1:
                    n_samples = info['n_samples']
                    break
            except Exception as e:
                warnings.warn(str(e))
            n_eps += 1
        vals.append(n_samples)
        print(vals)
    exit(0)
    # # # #
    # save((domain, problem, info), 'stuff')
    # exit(0)
    # domain, problem, info = load('stuff')

    problem.set_metric(None)

    n_planners = 4

    planners = [mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
                     mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
                     port=2323 + i,
                     wsl=False) for i in range(n_planners)]

    valid, output = plan_parallel(planners, domain, problem, n_attempts=10, verbose=True)

    if not valid:
        print("An error occurred :(")
        print(output)
    elif not output.valid:
        print("Planner could not find a valid plan :(")
        print(output.raw_output)
    else:
        print("We found a plan!")
        # get the plan out
        print(output.raw_output)
        print(output.path)
        make_video(env, domain, output.path)
        make_video(env, domain, output.path, pcas=[pca, pca2], name='pcas')

    exit(0)

    prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    start_link = int(prop.name[prop.name.index('_') + 1:])

    tasks = discover_hddl_tasks(domain, problem, start_link, verbose=True, draw=False, subgoal_method='voterank')
    hddl = HDDLDomain(domain)
    # str(hddl)
    # count = 0
    for task in tasks:
        hddl.add_task(task)

        # if count > 10:
        #     break
        #
        # if "Level-2" in task.name:
        #     hddl.add_task(task)
        #     count +=1

    save(hddl, make_path(save_dir, 'hddl_domain.pkl'))
    save(hddl, make_path(save_dir, 'domain.hddl'), binary=False)
    flat_domain = hddl.to_pddl()
    save(flat_domain, make_path(save_dir, 'flat_domain.pkl'))
    save(flat_domain, make_path(save_dir, 'flat_domain.hddl'), binary=False)

    exit(0)

    print("PLANNING!")

    planner = mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
                   mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
                   wsl=False,
                   max_time=30)

    n_retries = 10
    for _ in range(n_retries):
        valid, output = planner.find_plan(flat_domain, problem, verbose=True)

        if not valid:
            print("An error occurred :(")
            print(output)
            break
        elif not output.valid:
            print("Planner could not find a valid plan :(")
            print(output.raw_output)
        else:
            print("We found a plan!")
            # get the plan out
            print(output.raw_output)
            print(output.path)

            make_video(env, domain, output.path, directory=save_dir)
            break

    # print(hddl)
