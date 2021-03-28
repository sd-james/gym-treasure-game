import pygame

from gym_multi_treasure_game.envs.configs import CONFIG
from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from gym_multi_treasure_game.envs.vis_hierarchy import visualise_hierarchy
from gym_multi_treasure_game.utils import plan_parallel, make_video
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.clause import Clause
from pyddl.pddl.predicate import Predicate
from s2s.core.build_model import build_model
from s2s.core.build_pddl import _overlapping_dists
from s2s.env.s2s_env import View
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import make_dir, make_path, save, load, indent

if __name__ == '__main__':

    TASK = 1
    save_dir = '../data.bak/{}'.format(TASK)
    #
    # visualise_hierarchy(TASK, load(make_path(save_dir, 'hddl_domain.pkl')))
    #
    # exit(0)

    pca = PCA(PCA_STATE)

    pca.load('pca/models/dropped_key_pca_state.dat')

    pca2 = PCA(PCA_INVENTORY)

    pca2.load('pca/models/dropped_key_pca_inventory.dat')

    # env = MultiTreasureGame(TASK, pcas=[pca, pca2], split_inventory=True)
    # env.render()
    # surf = env.drawer.draw_background_to_surface()
    # pygame.image.save(surf, 'background.png')

    env = PCAWrapper(MultiTreasureGame(TASK, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)

    make_dir(save_dir, clean=False)

    # domain, problem, info = build_model(env,
    #                                     reload=True,
    #                                     save_dir=save_dir,
    #                                     # load_existing_data='/media/hdd/treasure_data/{}'.format(TASK),
    #                                     n_jobs=16,
    #                                     seed=0,
    #                                     n_episodes=50,
    #                                     options_per_episode=1000,
    #                                     view=View.AGENT,
    #                                     **CONFIG[TASK],
    #                                     visualise=True,
    #                                     verbose=True)
    #
    # domain = load(make_path(save_dir, 'linked_domain.pkl'))
    # problem = load(make_path(save_dir, 'linked_problem.pkl'))

    # import matplotlib.pyplot as plt
    #
    # env.reset()
    # for pred in domain.predicates:
    #     if pred != Predicate.not_failed() and pred.mask == [1]:
    #         data = pred.sample(100)
    #         import numpy as np
    #
    #         # print(np.mean(data, axis=0))
    #         print(np.mean(data))
    #
    #         data = np.array(
    #             [0.15298427095921896, 0.7495818600191144, 9.889323706790858, -9.034375041323432, 7.170734195078953]
    #             )
    #         data = np.expand_dims(data, axis=0)
    #         data = np.expand_dims(data, axis=0)
    #         im = env.render_states(data, mask=pred.mask, masked=True, view=View.AGENT)
    #         plt.imshow(im)
    #         plt.show()
    #
    # exit(0)
    #
    # # problem.set_metric(None)
    # domain.probabilistic = False
    #
    # print(domain)
    # print(problem)
    # #
    # n_planners = 1
    # planners = [mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
    #                  mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
    #                  port=2323 + i,
    #                  wsl=False) for i in range(n_planners)]
    #
    # valid, output = plan_parallel(planners, domain, problem, n_attempts=10, verbose=True)
    #
    # if not valid:
    #     print("An error occurred :(")
    #     print(output)
    # elif not output.valid:
    #     print("Planner could not find a valid plan :(")
    #     print(output.raw_output)
    # else:
    #     print("We found a plan!")
    #     # get the plan out
    #     print(output.raw_output)
    #     print(output.path)
    #     make_video(env, domain, output.path)
    #     # make_video(env, domain, output.path, pcas=[pca, pca2], name='pcas')
    #
    # exit(0)
    #
    domain = load(make_path(save_dir, 'linked_domain.pkl'))
    problem = load(make_path(save_dir, 'linked_problem.pkl'))

    prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    start_link = int(prop.name[prop.name.index('_') + 1:])

    tasks = discover_hddl_tasks(domain, verbose=True, draw=False, subgoal_method='voterank', problem=problem,
                                initial_link=start_link, reduce_graph=3)

    for task in tasks:
        if 'Level-3' in task.name:
            print(task.name)
            print()
            for method in task.methods:
                operator = method.flatten()
                print(indent(method))
                print(indent(operator.data['options']))
                print()

    exit(0)

    hddl = HDDLDomain(domain)
    # count = 0
    for task in tasks:
        hddl.add_task(task)
    print(hddl)

    # exit(0)

    visualise_hierarchy(TASK, hddl, [pca, pca2])

    # if count > 10:
    #     break
    #
    # if "Level-2" in task.name:
    #     hddl.add_task(task)
    #     count +=1

    # save(hddl, make_path(save_dir, 'hddl_domain.pkl'))
    # save(hddl, make_path(save_dir, 'domain.hddl'), binary=False)
    flat_domain = hddl.to_pddl()
    # save(flat_domain, make_path(save_dir, 'flat_domain.pkl'))
    # save(flat_domain, make_path(save_dir, 'flat_domain.hddl'), binary=False)

    # exit(0)

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
