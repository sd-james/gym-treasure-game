from functools import partial

import numpy as np

from gym_multi_treasure_game.exps.graph_utils import shortest_path_edges, find_nodes, extract_options
from s2s.utils import load, run_parallel


def find_plan(graph, source, target):
    """
    Return the set of options for the shortest path from source to target, or None if no path exists
    """
    edges = shortest_path_edges(graph, source, target)
    if edges is None:
        return None
    plan = list()
    for edge in edges:
        plan += extract_options(edge)
    return plan


def is_match(graph, sources, targets, true_plan):
    """
    Determine if there exists a plan that gets from one of the sources to one of the targets and is the same as
    the true plan
    """
    for source in sources:
        for target in targets:
            plan = find_plan(graph, source, target)
            if plan is not None and plan == true_plan:
                return True
    return False


def find_mapping(graph, truth, node):
    """
    Find the nodes in the current graph that correspond to the node in the ground truth graph
    """
    return find_nodes(truth.nodes[node]['state'], graph)


def _evaluate_plans(truth, graph, test_cases):
    count = 0
    for start, plan, end in test_cases:

        sources = find_mapping(graph, truth, start)
        targets = find_mapping(graph, truth, end)

        if is_match(graph, sources, targets, plan):
            count += 1
    return count

def evaluate_plans(task, truth, graph, n_jobs=1):
    test_cases = load('data/test_cases.pkl')[task]
    splits = np.array_split(test_cases, n_jobs)
    functions = [partial(_evaluate_plans, truth, graph, split) for split in splits]
    ret = run_parallel(functions)
    count = sum(ret)
    return count / len(test_cases)


if __name__ == '__main__':
    import networkx as nx

    truth = nx.read_gpickle(
        '/home/steve/PycharmProjects/gym-multi-treasure-game/gym_multi_treasure_game/exps/data/ground_truth/graph_1.pkl')
    graph = nx.read_gpickle('/media/hdd/treasure_data/1/0/50/edge_info_graph_0_1_50.pkl')

    # draw(truth, True)
    # draw(graph, False)

    print(evaluate_plans(1, truth, graph, n_jobs=20))


def merge_graphs(A, B):
    pass
