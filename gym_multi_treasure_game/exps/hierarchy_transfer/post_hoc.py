import random

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from gym_multi_treasure_game.exps.graph_utils import merge, clean_and_fit, merge_and_clean
from gym_multi_treasure_game.exps.hierarchy_transfer.generate_hierarchy import compute_hierarchical_graph
from gym_multi_treasure_game.exps.transfer.plot_new_transfer import load_data
from s2s.utils import make_path, save, now, exists


def process(data):
    records = list()
    previous_graph = None
    classifiers = dict()
    dir = '/media/hdd/treasure_data'
    for task_count, (task, values) in tqdm(enumerate(data.items())):
        n_episodes = values[-1][1]
        experiment = values[-1][0]
        save_dir = make_path(dir, task, experiment, n_episodes)

        graph_path = make_path(save_dir, "pred_edge_info_graph_{}_{}_{}.pkl".format(experiment, task, n_episodes))
        assert exists(graph_path)
        transition_data = pd.read_pickle(make_path(save_dir, "transition.pkl"), compression='gzip')

        graph = nx.read_gpickle(graph_path)
        graph = compute_hierarchical_graph(graph, max_length=4, reduce_graph=3, subgoal_method='voterank')
        original_graph = graph.copy()

        graph, clusterer, to_keep, saved_hierarchy = merge(graph, previous_graph, transition_data, classifiers,
                                                           n_jobs=20)
        # if task_count == 0:
        #     raw_score = evaluate_plans(test_cases[task], ground_truth, graph, clusterer, n_jobs=20)
        #     print("{} vs {}".format(raw_score, values[-1][]))

        records.append((graph.copy(), saved_hierarchy))

        print("Merging...")
        time = now()
        classifiers = clean_and_fit(classifiers, task, original_graph)
        previous_graph = merge_and_clean(previous_graph, original_graph, task, classifiers)
        print('Merging took {} ms'.format(now() - time))
    return records


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    length = 3
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    total_data = load_data('../data/transfer_results', with_file=True)
    record = list()
    for file, data in tqdm(total_data):
        x = int(file[file.rindex('_') + 1:file.rindex('.')])
        if x != seed:
            continue
        record.append(process(data))
        save(record, 'hierarchy.pkl')
