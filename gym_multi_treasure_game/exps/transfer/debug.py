import networkx as nx

from gym_multi_treasure_game.exps.eval2 import evaluate_n_step
from gym_multi_treasure_game.exps.eval3 import new_evaluate
from gym_multi_treasure_game.exps.transfer.test_transfer import draw

if __name__ == '__main__':
    truth = '../data/ground_truth/graph_2.pkl'
    no_transfer = '/media/hdd/treasure_data/2/2/30/info_graph_2_2_30.pkl'
    transfer = 'test.gpickle'

    x = nx.read_gpickle('/media/hdd/treasure_data/1/0/30/edge_info_graph_0_1_30.pkl')

    # truth = nx.read_gpickle('edge_info_graph_0_1_1.pkl')

    truth = nx.read_gpickle(truth)
    no_transfer = nx.read_gpickle(no_transfer)
    transfer = nx.read_gpickle(transfer)

    # draw(truth, True)
    # draw(no_transfer, False)
    # draw(transfer, False)

    for graph in [no_transfer, transfer]:
        # for graph in [transfer]:
        # raw_score = evaluate_n_step(truth, graph, n=3, get_stats=False)
        # print(raw_score)
        raw_score = new_evaluate(truth, graph, get_stats=False)
        print(raw_score)
