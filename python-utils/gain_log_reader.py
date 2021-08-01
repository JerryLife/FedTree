import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt


def read_log(path):
    all_gains = []
    gains_per_tree = []
    with open(path, 'r') as f:
        print("Start loading")
        for line in f:
            if line == '\n':
                pass
            elif 'Tree' in line:
                all_gains.append(gains_per_tree)
                gains_per_tree = []
                print("Tree {} finished loading".format(line.split()[-1]))
            else:
                gains_per_tree.append([float(x) for x in line.split(',') if x.strip()])

    return all_gains


def plot_gains(gains, output_dir):
    """
    :param gains: [List of trees [List of nodes [List of gains]]]
    :return: void
    """
    for i, tree in enumerate(gains):
        for j, node_gains in enumerate(tree):
            top_node_gains = list(sorted(node_gains, reverse=True))[:100]
            plt.bar(np.arange(len(top_node_gains)), top_node_gains, width=0.7)
            plt.title("Top 100 gains on tree {} node {}\n".format(i, j) +
                      "This node contains {} valid gains in total".format(len(node_gains)))
            plt.ylabel("Gain (w/o multiplying learning rate)")
            plt.xlabel("Node indices")

            plt.savefig(os.path.join(output_dir, "tree_{}_node_{}.png".format(i, j)))
            plt.close()
        print("Tree {} plotted".format(i))


if __name__ == '__main__':
    gains = read_log("../log/gain.csv")
    plot_gains(gains, "../fig/gains/")
