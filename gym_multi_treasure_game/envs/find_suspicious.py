import os

from s2s.utils import exists


def make_path(root,
              *args):
    """
    Creates a path from the given parameters
    :param root: the root of the path
    :param args: the elements of the path
    :return: a string, each element separated by a forward slash.
    """
    path = root
    if path.endswith('/'):
        path = path[0:-1]
    for element in args:
        if not isinstance(element, str):
            element = str(element)
        if element[0] != '/':
            path += '/'
        path += element
    return path


if __name__ == '__main__':

    dir = '/media/hdd/treasure_data'
    tot_scores = []
    for task in range(1, 11):
        for n_samples in range(1, 51):
            for exp in range(10):
                for exp2 in range(exp + 1, 5):
                    A = make_path(dir, task, exp, n_samples, 'transition.pkl')
                    B = make_path(dir, task, exp2, n_samples, 'transition.pkl')

                    if not exists(A) or not exists(B):
                        continue

                    a = os.path.getsize(A)
                    b = os.path.getsize(B)
                    if a == b:
                        print("{} = {}".format(A, B))
