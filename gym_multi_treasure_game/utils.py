import random
from functools import partial
from typing import List

import cv2
import numpy as np
import pandas as pd

from gym_multi_treasure_game.envs.multi_treasure_game import MultiTreasureGame
from pyddl.pddl.domain import Domain
from s2s.planner.mgpt_planner import PlanOutput
from s2s.utils import make_path, run_parallel, plan, exists


def _make_video(directory: str, env, name: str, frames):
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
    _make_video(directory, env, '{}-{}'.format(env.name, name), global_views)


def plan_parallel(planners, domain, problem, n_attempts=10, verbose=False):
    functions = [partial(plan, planner, domain, problem, n_attempts=n_attempts, verbose=verbose) for planner in
                 planners]
    results = run_parallel(functions)
    error = not any(x for x, _ in results)
    found = any(isinstance(x, PlanOutput) and x.valid for _, x in results)
    best_output = None
    for valid, output in results:
        if error:
            return valid, output
        elif not found:
            return valid, output
        elif valid and output.valid:
            if best_output is None or len(best_output.path) > len(output.path):
                best_output = output
    return True, best_output


if __name__ == '__main__':
    transitions, init = extract_random_episodes('/media/hdd/treasure_data/1', 10)
