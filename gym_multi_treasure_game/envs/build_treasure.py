from typing import List

import cv2
import imageio
import numpy as np

from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from gym_multi_treasure_game.envs.pca.base_pca import PCA_N
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
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import make_dir, make_path, save, load


def _make_video(directory: str, name: str, frames):
    height, width, layers = np.array(frames[0]).shape
    print("Writing to video {}".format(env.name))
    file = make_path(directory, name)
    imageio.mimsave('{}.gif'.format(file), frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('{}.mp4'.format(file), fourcc, 10, (width, height))
    for frame in frames:
        # writer.writeFrame(frame[:, :, ::-1])
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
    # writer.close()  # close the writer


def make_video(env, domain: Domain, path: List[str], directory='.') -> None:
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
    global_views, local_views = FourFourRoomsEnv.animate(env._version, plan)
    _make_video(directory, '{}-{}'.format(env.name, 'global'), global_views)
    _make_video(directory, '{}-{}'.format(env.name, 'local'), local_views)


if __name__ == '__main__':

    TASK = 1
    pca = PCA(PCA_N)

    pca.load('pca/models/20_runs_pca_30_empty_inv.dat')

    env = PCAWrapper(MultiTreasureGame(TASK, pca), pca)
    save_dir = '../data/{}'.format(TASK)
    make_dir(save_dir, clean=False)
    domain, problem, info = build_model(env,
                                        save_dir=save_dir,
                                        n_jobs=16,
                                        seed=0,
                                        n_episodes=30,
                                        options_per_episode=1000,
                                        view=View.AGENT,
                                        linking_threshold=0.05,
                                        specify_rewards=True,
                                        effect_epsilon=4,
                                        init_epsilon=4,
                                        augment_negative=True,
                                        max_precondition_samples=5000,
                                        precondition_c_range=np.logspace(0.01, 0.5, 10),
                                        precondition_gamma_range=np.logspace(0.1, 1, 10),
                                        visualise=True,
                                        verbose=True)
    #
    save((domain, problem, info), 'stuff')
    # exit(0)
    domain, problem, info = load('stuff')

    # problem.goal = Clause([Predicate.not_failed(), Predicate('psymbol_0')])

    planner = mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
                   mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
                   wsl=False)

    n_retries = 1
    for _ in range(n_retries):
        valid, output = planner.find_plan(domain, problem)

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
            break

    exit(-1)

    prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    start_link = int(prop.name[prop.name.index('_') + 1:])

    tasks = discover_hddl_tasks(domain, problem, start_link, verbose=False, draw=True)
    hddl = HDDLDomain(domain)
    for task in tasks:
        hddl.add_task(task)

    save(hddl, make_path(save_dir, 'domain.hddl'), binary=False)
    flat_domain = hddl.to_pddl()
    save(flat_domain, make_path(save_dir, 'flat_domain.hddl'), binary=False)

    planner = mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
                   mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
                   wsl=False)
    n_retries = 10
    for _ in range(n_retries):
        valid, output = planner.find_plan(flat_domain, problem)

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

    print(hddl)
