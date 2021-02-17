from functools import partial
from typing import List
import matplotlib.pyplot as plt

from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gym_multi_treasure_game.utils import plan_parallel
from pyddl.pddl.domain import Domain
from pyddl.pddl.operator import Operator
from pyddl.pddl.predicate import Predicate
from s2s.hierarchy.discover_hddl_methods import _generate_mdp, _generate_full_mdp
from s2s.hierarchy.network import Node, Path
from s2s.hierarchy.option_discovery import _get_path
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.problem_symbols import _ProblemProposition
from s2s.utils import save, run_parallel


def execute_operators(env, plan: List[Operator]):
    def _get_options(operator):
        if 'options' in operator.data:
            return operator.data['options']
        return [operator.option]

    env.reset()
    reward = 0
    done = False
    for operator in plan:
        for option in _get_options(operator):
            _, _, r, done, _ = env.step(option)
            # env.render()
            if r is not None:
                reward += r
            if done:
                break
    env.close()
    return done, reward


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
    return execute_operators(env, plan)


def _combine(state, inv):
    has_key = 0
    has_gold = 0

    if inv is not None:
        m = np.mean(inv)
        if -1.6 <= m <= -1.5:
            pass
        elif 0.7 <= m <= 0.8:
            has_key = 1
        elif 3.7 <= m <= 3.8:
            has_key = 1
        elif -0.7 <= m <= -0.5:
            has_key = 1
            has_gold = 1
        elif 1.5 <= m <= 1.6:
            has_key = 1
            has_gold = 1
        elif m >= 1.35:
            has_key = 1
            has_gold = 1
        # else:
        #     raise ValueError

    return np.hstack((state, [has_key, has_gold]))


def _count_paths(predicted, actual, candidates):
    def _dist(a, b):
        m = min(len(a), len(b))
        a = a[0:m]
        b = b[0:m]
        for i in range(len(a)):
            if abs(a[i] - b[i]) > 0.09:
                return np.inf
        return np.linalg.norm(a - b)

    def _find_node(state, graph):

        if len(state.shape) == 2:
            state = state.squeeze()
        best_node = None
        best_dist = np.inf
        for node in graph.nodes:
            s = graph.nodes[node]['state']
            if len(s.shape) == 2:
                s = s.squeeze()
            s = _combine(s, graph.nodes[node]['obs'][1])
            dist = _dist(s, state)
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node

    find = 0

    for start_node, end_node in candidates:
        source = _find_node(actual.nodes[start_node]['state'], predicted)
        if source is None:
            continue
        target = _find_node(actual.nodes[end_node]['state'], predicted)
        if target is not None and nx.has_path(predicted, source, target):
            find += 1
    return find, len(candidates)


def _score_reachable(predicted, actual, **kwargs):
    n_jobs = kwargs.get('n_jobs', 1)

    candidates = list()
    for start_node in actual.nodes:
        for end_node in actual.nodes:
            if start_node == end_node or not nx.has_path(actual, start_node, end_node):
                continue
            candidates.append((start_node, end_node))

    splits = np.array_split(candidates, n_jobs)
    functions = [partial(_count_paths, predicted, actual, splits[i]) for i in range(n_jobs)]
    results = run_parallel(functions)
    find = 0
    count = 0
    for f, t in results:
        find += f
        count += t
    return find / count


def _score(predicted, actual):
    def _node_match(a, b):
        a = a['state'].squeeze()
        b = b['state'][0:len(a)]
        for i in range(len(a)):
            if abs(a[i] - b[i]) > 0.09:
                return False
        return True

    score = next(nx.optimize_graph_edit_distance(predicted, actual))
    print("Edit dist: {}".format(score))
    score = next(nx.optimize_graph_edit_distance(predicted, actual, node_match=_node_match))
    print("Edit dist with node match: {}".format(score))
    score = next(
        nx.optimize_graph_edit_distance(predicted, actual, node_match=_node_match, node_ins_cost=lambda x: 100))
    print("Edit dist with node match and penalty: {}".format(score))
    return score


def _sample(state):
    ground_state = [None] * 2
    for pred in state.predicates:
        data = np.mean(pred.sample(100), axis=0)
        for i, m in enumerate(pred.mask):
            ground_state[m] = data[i]
    return ground_state


def __get_pos(pos, obs):
    mean = _combine(pos, obs)
    pos = np.array([mean[0], 1 - mean[1]])
    if mean[2] == 1:
        pos += [0.01, -0.02]
    if len(mean) >= 4 and mean[3] == 1:
        pos += [0.03, 0.03]
    if len(mean) >= 5 and mean[4] == 1:
        pos += [-0.02, -0.02]
    return pos


def _op_length(operator):
    if 'options' in operator.data:
        return len(operator.data['options'])
    return 1


def _op_level(operator):
    level = 0
    while 'operators' in operator.data:
        level += 1
        operator = operator.data['operators'][0]
    return level


def draw(graph, ground_truth, show=True):
    positions = dict()
    if ground_truth:
        for node in graph.nodes:
            positions[node] = graph.nodes[node]['pos']
    else:
        for node in graph.nodes:
            problem_data = graph.nodes[node]['state']
            obs_data = graph.nodes[node]['obs']
            positions[node] = __get_pos(problem_data, obs_data[1])
    nx.draw(graph, pos=positions)
    if show:
        plt.show()




def evaluate_similarity_pretrained(ground_truth, graph, **kwargs):
    if kwargs.get('draw', False):
        draw(graph, False)
        draw(ground_truth, True)
    return _score_reachable(graph, ground_truth, **kwargs), graph


def evaluate_similarity(ground_truth, domain, **kwargs):
    portable_symbols = [x for x in domain.predicates if x != Predicate.not_failed() and 'psymbol' not in x.name]
    problem_symbols = [x for x in domain.predicates if x != Predicate.not_failed() and 'psymbol' in x.name]

    states, transitions = _generate_full_mdp(portable_symbols, problem_symbols, domain.linked_operators)

    graph = nx.DiGraph()
    for start, edges in transitions.items():
        for edge in edges:
            end = edge.end_node
            graph.add_edge(start.id, end.id, weight=edge.prob, length=_op_length(edge.action),
                           level=_op_level(edge.action))

    temp = dict()
    for predicate in domain.predicates:
        if isinstance(predicate, _ProblemProposition):
            link = int(predicate.name[predicate.name.index('_') + 1:])
            temp[link] = predicate
    positions = dict()
    for state in states:
        symbol = temp[state.link]
        problem_data = symbol.sample(1)[0]
        obs_data = _sample(state)
        # x, y = tuple(problem_data[0:2] + 0.5 * np.random.normal(0, 0.05, 2))
        # positions[state.id] = (x, 1 - y)
        positions[state.id] = __get_pos(problem_data, obs_data[1])
        graph.nodes[state.id]['state'] = problem_data
        graph.nodes[state.id]['obs'] = obs_data

    if kwargs.get('draw', False):
        nx.draw(graph, pos=positions)
        plt.show()

    if kwargs.get('draw', False):
        draw(ground_truth, True)
    # _score(graph, ground_truth)
    return _score_reachable(graph, ground_truth, **kwargs), graph


def evaluate_with_network(domain, problem):
    prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    start_link = int(prop.name[prop.name.index('_') + 1:])

    states, transitions = _generate_mdp(problem.init, domain.linked_operators, start_link)

    graph = nx.DiGraph()
    for start, edges in transitions.items():
        for edge in edges:
            end = edge.end_node
            graph.add_edge(start.id, end.id, weight=-edge.reward * edge.prob)

    # A = np.zeros((len(states), len(states)))
    # for start, edges in transitions.items():
    #     for edge in edges:
    #         end = edge.end_node
    #         A[start.id, end.id] = 1
    #         assert start.id == start.state.id
    #         assert end.id == end.state.id
    # graph = nx.from_numpy_array(A, create_using=nx.DiGraph)
    goal = _extract_goal(states, problem.goal)
    if goal is None:
        return 0, []
    goal = goal.state.id

    path = nx.shortest_path(graph, 0, goal)
    # path = nx.shortest_path(graph, 0, goal, weight='weight')
    path = _get_path(transitions, path)
    return nx.has_path(graph, 0, goal), path


def validate(env, path: Path):
    """
    Create a video of the agent solving the task
    :param version_number: the environment number
    :param domain: the PDDL domain
    :param path: the list of PDDL operators to execute
    :param directory: the directory where the video should be written to
    """
    plan = [x.action for x in path.walk()]
    print(plan)
    done, reward = execute_operators(env, plan)
    return int(done)


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


def _dfs(state, transitions, plan, goal):
    if goal is None:
        return 0
    if len(plan) == 0:
        if state == goal:
            return 1
    action = plan[0]
    max_prob = 0
    for edge in transitions[state]:
        if edge.action.option == action:
            prob = _dfs(edge.end_node, transitions, plan[1:], goal)
            max_prob = max(max_prob, prob * edge.prob)
            if max_prob == 1:
                break
    return max_prob


def _extract_goal(states, goal):
    names = {g.name for g in goal if g != Predicate.not_failed()}
    for state in states:
        other = {x.name for x in state.predicates if x != Predicate.not_failed()}
        other.add('psymbol_{}'.format(state.link))
        if names.issubset(other):
            return Node(state)
    return None


def evaluate_manually(domain, problem, true_plan):
    prop = next(x for x in problem.init if isinstance(x, _ProblemProposition))
    start_link = int(prop.name[prop.name.index('_') + 1:])
    states, transitions = _generate_mdp(problem.init, domain.linked_operators, start_link)
    curr = states[0]
    prob = _dfs(Node(curr), transitions, true_plan, _extract_goal(states, problem.goal))
    return prob > 0
