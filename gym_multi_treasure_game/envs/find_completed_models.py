import matplotlib.pyplot as plt
import numpy as np

from s2s.utils import make_path, load, range_without, exists

if __name__ == '__main__':

    dir = '/media/hdd/treasure_data/'

    miss = list()
    for i in range(5):
        for task in range(1, 11):

            miss = list()
            for n in range(1, 51):
                path = make_path(dir, task, i, n)
                # if not exists(make_path(path, 'effects.pkl')):
                if not exists(make_path(path, 'info_graph_{}_{}_{}.pkl'.format(i, task, n))):
                    miss.append(n)

            if len(miss) > 0:
                prog = 100 * len(miss) / 50
                print("Missing task {}, exp {}: {}%".format(task, i, prog))
                print("\t{}".format(' '.join(map(str, miss))))

