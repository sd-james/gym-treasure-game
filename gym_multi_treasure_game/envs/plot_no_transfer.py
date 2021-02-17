import matplotlib.pyplot as plt
import numpy as np

from s2s.utils import make_path, load, range_without, exists


def extract_scores(recorder, task, n_exps, n_samples):
    dir = '/media/hdd/full_treasure_data/treasure_data'
    scores = list()
    samples = list()
    for exp in range(n_exps):
        score = list()
        sample = list()
        for n in range(1, 51):
            score.append(recorder.get_score(exp, task, n))
            sample.append(n_samples[(exp, task, n)])
        scores.append(score)
        samples.append(sample)
    return np.mean(samples, axis=0), np.mean(scores, axis=0), np.std(scores, axis=0)



if __name__ == '__main__':

    dir = '/media/hdd/treasure_data/'
    recorder = load('no_transfer_results.pkl')
    tot_scores = []
    for task in range_without(1, 11):
        for exp in range(5):
            scores = list()
            for n_samples in range(1, 51):
                score = recorder.get_score(exp, (task - 1, task), n_samples)
                scores.append(score)
    means = np.mean(tot_scores, axis=0)
    dev = np.std(tot_scores, axis=0)
    plt.plot(np.arange(1, len(means) + 1), means)
    plt.fill_between(np.arange(1, len(means) + 1), means - dev, means + dev, alpha=0.5)

    plt.show()
