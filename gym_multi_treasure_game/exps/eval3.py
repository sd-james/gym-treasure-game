import itertools

import networkx as nx

from gym_multi_treasure_game.exps.eval2 import _find_mapping


def get_plan(graph, path):
    plan = list()
    for i in range(len(path) - 1):
        options = graph.edges[(path[i], path[i + 1])]['options']
        if len(set(options)) > 1:
            raise ValueError
        plan.append(options[0])
        # plan.append(list(set(options)))
    return plan



def _is_match(mapping, shortest, starts, ends, true_path, use_hierarchy=False):
    for x, y in itertools.product(starts, ends):
        if not x in shortest or y not in shortest[x]:
            continue

        pred_path = shortest[x][y]
        if len(pred_path) == len(true_path):
            return True
    return False


    #     if len(pred_path) > len(true_path):
    #         continue
    #     if use_hierarchy and len(pred_path) < len(true_path):
    #
    #         to_match = [mapping[x] for x in true_path]
    #         candidates = [_get_idx(x, to_match) for x in pred_path]
    #
    #         if not any(len(x) == 0 for x in candidates):
    #             return True  # TODO approximation!
    #
    #         # if is_increasing(canidates):
    #         #     # if there is a path
    #         #     return True
    #
    #     else:
    #         if len(pred_path) != len(true_path):
    #             continue
    #         match = True
    #         for true, pred in zip(true_path, pred_path):
    #             if pred not in mapping[true]:
    #                 match = False
    #                 break
    #         if match:
    #             return True
    #
    # return False

def new_evaluate(ground_truth, graph, get_stats=False, use_hierarchy=False):
    paths = dict(nx.all_pairs_shortest_path(ground_truth, cutoff=None))
    mapping = _find_mapping(graph, ground_truth)
    shortest = dict(nx.all_pairs_shortest_path(graph), cutoff=None)
    count = 0
    find = 0
    stats_B = list()
    for start, items in paths.items():
        A = mapping[start]
        for end, path in items.items():
            if start == end:
                continue
            B = mapping[end]
            count += 1

            # true_plan = get_plan(ground_truth, path)

            if _is_match(mapping, shortest, A, B, path, use_hierarchy=use_hierarchy):
                # print(count)
                find += 1
                if get_stats:
                    stats_B.append(len(path))
    if get_stats:
        return find, stats_B
    return find
