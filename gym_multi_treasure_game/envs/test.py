from gym_multi_treasure_game.envs.build_treasure import plan_parallel
from pyddl.pddl.clause import Clause
from pyddl.pddl.predicate import Predicate
from s2s.planner.mgpt_planner import mGPT
from s2s.utils import load

if __name__ == '__main__':

    TASK = 2
    domain = load('../data/{}/linked_domain.pkl'.format(TASK))
    problem = load('../data/{}/linked_problem.pkl'.format(TASK))

    problem.set_metric(None)
    problem.goal = Clause([Predicate.not_failed(), Predicate('symbol_21')])

    n_planners = 1

    planners = [mGPT(mdpsim_path='../../../hierarchical-skills-to-symbols/s2s/planner/mdpsim-1.23/mdpsim',
                     mgpt_path='../../../hierarchical-skills-to-symbols/s2s/planner/mgpt/planner',
                     port=9999 + i,
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
