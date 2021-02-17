import matplotlib.pyplot as plt
import numpy as np

from s2s.utils import make_path, load, range_without, exists

if __name__ == '__main__':

    dir = '/media/hdd/treasure_data/'

    miss = list()
    for i in range(5):
        for task in range(1, 11):
            for n in range(1, 51):
                path = make_path(dir, task, i, n)
                # if not exists(make_path(path, 'effects.pkl')):
                if not exists(make_path(path, 'info_graph_{}_{}_{}.pkl'.format(i, task, n))):
                    miss.append(path)

    for m in miss:
        print("Missing {}".format(m))
    exit(0)

    dir = '/media/hdd/full_treasure_data/treasure_data'

    for task in range_without(1, 11, 3):
        tot_scores = []
        for exp in range(1):

            recorders = load(make_path(dir, 'recorders_{}_{}_1_50.pkl'.format(exp, task)))
            recorder = recorders[0]

            scores = list()
            for n_samples in range(1, 51):
                score = recorder.get_score(exp, task, n_samples)
                scores.append(score)
            tot_scores.append(scores)
        means = np.mean(tot_scores, axis=0)
        dev = np.std(tot_scores, axis=0)
        plt.plot(np.arange(1, len(means) + 1), means)
        plt.fill_between(np.arange(1, len(means) + 1), means - dev, means + dev, alpha=0.5)

        plt.show()
