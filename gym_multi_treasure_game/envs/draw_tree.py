from collections import deque

import cv2
import networkx as nx
from PIL import Image
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

from s2s.utils import make_path


def path(name):
    return make_path('/media/hdd/Documents/Research/PhD/Thesis/res/images/hierarchy/gifs', name)


def image(name):
    return Image.open(path(name + ".png"))


class Node:

    def __init__(self, name, children=None):
        self.name = name
        if children is None:
            children = list()
        self.children = children


def A():
    print("A")
    return True


def B():
    print("B")
    return True


def C():
    print("C")
    return False


if __name__ == '__main__':

    root = Node('a', [Node('_a_1'), Node('_a_2'), Node('_a_3'), Node('_a_4')])

    G = nx.DiGraph()
    G.add_node(root.name)

    queue = deque()
    queue.append(root)
    while len(queue) > 0:
        current = queue.popleft()
        G.add_node(current.name)
        for child in current.children:
            queue.append(child)
            G.add_node(child.name)
            G.add_edge(current.name, child.name)
    # G.add_node("ROOT")
    #
    # for i in range(5):
    #     G.add_node("Child_%i" % i)
    #     G.add_node("Grandchild_%i" % i)
    #     G.add_node("Greatgrandchild_%i" % i)
    #
    #     G.add_edge("ROOT", "Child_%i" % i)
    #     G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
    #     G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)

    # write dot file to use with graphviz
    # run "dot -Tpng test.dot >test.png"
    # write_dot(G, 'test.dot')

    # same layout using matplotlib with no labels
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=5000)

    ax = plt.gca()
    fig = plt.gcf()
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    imsize = 0.5  # this is the image size
    # for n in G.nodes():
    #     (x, y) = pos[n]
    #     xx, yy = trans((x, y))  # figure coordinates
    #     xa, ya = trans2((xx, yy))  # axes coordinates
    #     a = plt.axes([xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize])
    #     a.imshow(image(n))
    #     a.set_aspect('equal')
    #     a.axis('off')

    plt.show()
    # plt.savefig('nx_test.png')
