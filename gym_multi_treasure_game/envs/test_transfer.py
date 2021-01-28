from typing import List

import cv2
import imageio
import numpy as np

from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_STATE, PCA_INVENTORY
from gym_multi_treasure_game.envs.pca.pca import PCA
from gym_multi_treasure_game.envs.pca.pca_wrapper import PCAWrapper
from pyddl.hddl.hddl_domain import HDDLDomain
from pyddl.pddl.clause import Clause
from pyddl.pddl.domain import Domain
from pyddl.pddl.predicate import Predicate
from s2s.core.build_model import build_model
from s2s.env.s2s_env import View
from s2s.env.walkthrough_env import FourFourRoomsEnv
from s2s.hierarchy.discover_hddl_methods import discover_hddl_tasks
from s2s.pddl.linked_operator import LinkedPDDLOperator
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.exploration import collect_data_with_existing
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import make_dir, make_path, save, load


def _make_video(directory: str, name: str, frames):
    height, width, layers = np.array(frames[0]).shape
    print("Writing to video {}".format(env.name))
    file = make_path(directory, name)
    # imageio.mimsave('{}.gif'.format(file), frames, fps=60)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('{}.mp4'.format(file), fourcc, 60, (width, height))
    for frame in frames:
        # writer.writeFrame(frame[:, :, ::-1])
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
    # writer.close()  # close the writer


def make_video(env, domain: Domain, path: List[str], directory='.', pcas=None, name='global') -> None:
    """
    Create a video of the agent solving the task
    :param version_number: the environment number
    :param domain: the PDDL domain
    :param path: the list of PDDL operators to execute
    :param directory: the directory where the video should be written to
    """

    plan = list()
    for option in path:
        operator = [x for x in domain.operators if x.name.lower() == option]
        assert len(operator) == 1
        operator = operator[0]
        plan.append(operator)

    # make video!!
    global_views = MultiTreasureGame.animate(env.version, pcas, plan)
    _make_video(directory, '{}-{}'.format(env.name, name), global_views)


def extract(domain):
    operators = set()
    for operator in domain.operators:
        parent = operator.data.get('parent', None)
        if parent is not None:
            operators.add(parent)
        elif 'operators' in operator.data:
            chain = operator.data['operators']
            operator.data['operators'] = [x.data['parent'] for x in chain]
            operators.add(LinkedPDDLOperator(operator))
        else:
            raise ValueError

    return [predicate for predicate in domain.predicates if predicate != Predicate.not_failed()
            and not isinstance(predicate, _ProblemProposition)], list(operators)


if __name__ == '__main__':

    TASK = 1
    pca = PCA(PCA_STATE)

    pca.load('pca/models/mod_no_fancy_pca_state.dat')

    pca2 = PCA(PCA_INVENTORY)

    pca2.load('pca/models/mod_no_fancy_pca_inventory.dat')

    env = PCAWrapper(MultiTreasureGame(TASK, pcas=[pca, pca2], split_inventory=True), pca, pca2=pca2)
    save_dir = '../data.bak/{}'.format(TASK)

    domain = load(make_path(save_dir, 'linked_domain.pkl'))
    problem = load(make_path(save_dir, 'linked_problem.pkl'))
    clusterer = load(make_path(save_dir, 'quick_cluster.pkl'))
    prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    start_link = int(prop.name[prop.name.index('_') + 1:])

    tasks = discover_hddl_tasks(domain, problem, start_link, verbose=True, draw=True, subgoal_method='voterank')
    hddl = HDDLDomain(domain)
    exit(0)
    #
    # for task in tasks:
    #     hddl.add_task(task)
    # save(hddl, make_path(save_dir, 'hddl_domain.pkl'))
    # save(hddl, make_path(save_dir, 'domain.hddl'), binary=False)
    # flat_domain = hddl.to_pddl()
    # save(flat_domain, make_path(save_dir, 'flat_domain.pkl'))
    #
    # exit(0)

    domain = load(make_path(save_dir, 'flat_domain.pkl'))

    previous_predicates, previous_operators = extract(domain)
    for op in previous_operators:
        op.clear()

    collect_data_with_existing(env, previous_predicates, previous_operators, max_timestep=np.inf, random_search=True,
                               max_episode=30, verbose=True, seed=None, n_jobs=1)

    done = False
    while not done:
        state, obs = env.reset()
        # print(state)
        for N in range(1000):
            mask = env.available_mask

            select_action(state, obs, mask, domain.operators)

            action = np.random.choice(np.arange(env.action_space.n), p=mask / mask.sum())
            next_state, next_obs, reward, done, info = env.step(action)
            # print(next_state)

            env.render('human', view=View.AGENT)
            if done:
                print("{}: WIN: {}".format(i, N))
                print(info)
                solved = True
                env.close()
                break
            time.sleep(0.5)

    # planner = mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
    #                mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
    #                wsl=False,
    #                max_time=30)
    #
    # n_retries = 10
    # for _ in range(n_retries):
    #     valid, output = planner.find_plan(domain, problem)
    #
    #     if not valid:
    #         print("An error occurred :(")
    #         print(output)
    #         break
    #     elif not output.valid:
    #         print("Planner could not find a valid plan :(")
    #         print(output.raw_output)
    #     else:
    #         print("We found a plan!")
    #         # get the plan out
    #         print(output.raw_output)
    #         print(output.path)
    #         make_video(env, domain, output.path)
    #         make_video(env, domain, output.path, pcas=[pca, pca2], name='pcas')
    #         break

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
