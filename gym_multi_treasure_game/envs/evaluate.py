from typing import List

from gym_multi_treasure_game.utils import plan_parallel
from pyddl.pddl.domain import Domain
from s2s.planner.mgpt_planner import mGPT
from s2s.utils import save


def validate_plan(env, domain: Domain, path: List[str]):
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

    def _get_options(operator):
        if 'options' in operator.data:
            return operator.data['options']
        return [operator.option]

    env.reset()
    reward = 0
    done = False
    for operator in plan:
        for option in _get_options(operator):
            _, r, done, _ = env.step(option)
            reward += r
            if done:
                break
    return done, reward


def evaluate_model(env, domain, problem, verbose=False):
    planners = [mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
                     mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
                     port=2323 + i,
                     wsl=False) for i in range(10)]

    save((domain, problem), 'uuuu')

    valid, output = plan_parallel(planners, domain, problem, n_attempts=10, verbose=verbose)
    if not valid:
        print("An error occurred :(")
        # print(output)
    elif not output.valid:
        print("Planner could not find a valid plan :(")
        # print(output.raw_output)
    else:
        print("We found a plan!")
        # get the plan out
        done, reward = validate_plan(env, domain, output.path)
        if done:
            return 1

    return 0
