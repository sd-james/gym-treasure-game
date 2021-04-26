import random
from collections import OrderedDict

import networkx as nx
import matplotlib.pyplot as plt


def get_graph():
    return nx.barbell_graph(4, 2, create_using=None)

if __name__ == '__main__':

    G = get_graph()
    # G.add_edges_from([(5,3)])
    # G.remove_edge(1,3)
    # G.remove_edge(0, 2)
    # G.remove_edge(9,8)
    # G.remove_edge(7,6)
    #
    # G.add_edges_from([(10, x) for x in [0,1,2,3]])
    # G.add_edges_from([(11, x) for x in [7,8,9,6]])
    # G.add_edges_from([(0,4), (8,5)])

    G2 = get_graph()
    nx.relabel_nodes(G2, {i: i + 10 for i in range(10)} , copy=False)

    G.update(G2)
    G.add_edges_from([(4, 14), (5, 15)])


    G.add_edges_from([(1, 20), (1, 21), (1, 22), (20,21), (20,22)])
    G.add_edges_from([(9, 23), (9, 24), (9, 25), (23,24), (23,25)])
    G.add_edges_from([(12, 26), (12, 27), (12, 28), (26,27), (26,28)])
    G.add_edges_from([(18, 29), (18, 30), (18, 31), (29,30), (29,31)])

    G.remove_edge(19,17)
    G.remove_edge(7,8)
    G.remove_edge(10,11)
    G.remove_edge(2,0)


    nx.draw(G, with_labels=True)
    plt.show()
    # nx.draw(G2, with_labels=True)
    # plt.show()

    ordered_subgoals = nx.algorithms.centrality.betweenness_centrality(G)
    ordered_subgoals = OrderedDict(sorted(ordered_subgoals.items(), key=lambda kv: kv[1], reverse=True))

    count = 0
    for x, v in ordered_subgoals.items():
        print(x, v)
        if count > 2:
            break
        count += 1

    print()
    targets = nx.algorithms.centrality.voterank(G, 4)
    for t in targets:
        print(t)

# G = nx.DiGraph()
#
# edges = [ (0, 2), (0, 6),  (1, 4), (2, 0), (2, 3), (2, 5), (2, 6),
#          (3, 4), (3, 2), (4, 3), (4, 5), (5, 2), (5, 4), (5, 6), (6, 0), (6, 2), (6, 5)]
#
# for i in range(7, 11):
#     for j in range(7, 11):
#         if i != j:
#             edges.append((i, j))
#
# edges2 = [(11, 12), (11, 13), (12, 11), (13, 11), (13, 16),  (14, 15), (14, 17), (15, 14), (15, 18),
#           (16, 13), (16, 17), (17, 16), (17, 14), (18, 15)]
#
# mid = [(0, 7), (4, 8), (11, 10), (16, 9), (16, 8)]
# mid += [(y, x) for x, y in mid]
#
# G.add_edges_from(edges + edges2 + mid)
#
#
# def get_pos(x):
#
#     if x < 7:
#         return random.randint(0, 5), random.randint(0,10)
#     if x < 11:
#         return random.randint(10, 15), random.randint(0,10)
#     return random.randint(20, 25), random.randint(0, 10)
#
# random.seed(1)
# pos = {x: get_pos(x) for x in range(19)}
#
# ordered_subgoals = nx.algorithms.centrality.betweenness_centrality(G)
# ordered_subgoals = OrderedDict(sorted(ordered_subgoals.items(), key=lambda kv: kv[1], reverse=True))
#
# for x, v in ordered_subgoals.items():
#     print(x, v)
#
# targets = nx.algorithms.centrality.voterank(G, 4)
# for t in targets:
#     print(t)
#
#
# nx.draw(G, pos=pos,with_labels=True)
#
# plt.show()
