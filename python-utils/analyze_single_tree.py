import json
from tqdm import tqdm
import numpy as np
from queue import Queue

from GBDT import Tree, GBDT, Node


class CompareResultPerLevel:
    def __init__(self):
        self.n_common_node = 0     # #nodes on the same place
        self.n_node_common_feature = 0      # #common nodes with the same split feature
        self.n_node_split_nbr_include = 0   # #common nodes s.t. split_bid2 falls within split_nbr1
        self.n_node_common_split_bid = 0    # #common nodes with the same split bid and split feature

    def add_common_node_(self, node1: Node, node2: Node):
        self.n_common_node += 1

        if node1.split_feature_id == node2.split_feature_id:
            self.n_node_common_feature += 1
            if node2.split_bid in range(*node1.split_nbr):
                self.n_node_split_nbr_include += 1
            if node1.split_bid == node2.split_bid:
                self.n_node_common_split_bid += 1


class CompareResult:
    def __init__(self, cr_level=None):
        self.cr_level = cr_level if cr_level is not None else []

    def add_common_node_(self, node1, node2, level):
        if level < len(self.cr_level):
            self.cr_level[level].add_common_node_(node1, node2)
        else:
            cr_per_level = CompareResultPerLevel()
            cr_per_level.add_common_node_(node1, node2)
            self.cr_level.append(cr_per_level)

    def n_common_node(self, level=None):    # sum n_common_node before level (excluding level)
        return sum([cr.n_common_node for cr in self.cr_level[:level]])

    def n_node_common_feature(self, level=None):
        return sum([cr.n_node_common_feature for cr in self.cr_level[:level]])

    def n_node_split_nbr_include(self, level=None):
        return sum([cr.n_node_split_nbr_include for cr in self.cr_level[:level]])

    def n_node_common_split_bid(self, level=None):
        return sum([cr.n_node_common_split_bid for cr in self.cr_level[:level]])


def compare_tree(tree1: Tree, tree2: Tree):
    cr = CompareResult()

    visit_queue1 = Queue()
    visit_queue2 = Queue()
    visit_queue1.put(0)
    visit_queue2.put(0)
    level = 0
    while visit_queue1.qsize() + visit_queue2.qsize() > 0:
        level_size1 = visit_queue1.qsize()
        level_size2 = visit_queue2.qsize()  # to keep track of the #nodes in a level
        while level_size1 > 0 or level_size2 > 0:
            node1_id = visit_queue1.get_nowait()
            node2_id = visit_queue2.get_nowait()
            node1 = tree1.nodes[node1_id]
            node2 = tree2.nodes[node2_id]

            # compare two nodes
            if not node1.is_leaf and not node2.is_leaf:
                cr.add_common_node_(node1, node2, level)
                visit_queue1.put(node1.lch_id)
                visit_queue1.put(node1.rch_id)
                visit_queue2.put(node2.lch_id)
                visit_queue2.put(node2.rch_id)

            level_size1 -= 1
            level_size2 -= 1
        level += 1
    return cr


if __name__ == '__main__':
    for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for dataset in ['codrna', 'cadata', 'covtype', 'msd', 'gisette']:
            model_path = f"../cache/{dataset}_single_tree_deltaboost.json"
            model_remain_path = f"../cache/alpha_{alpha:.1f}/{dataset}_single_tree_remain_deltaboost.json"
            with open(model_path, 'r') as f1, open(model_remain_path, 'r') as f2:
                js1 = json.load(f1)
                js2 = json.load(f2)
            gbdt = GBDT.load_from_json(js1, 'deltaboost')
            gbdt_remain = GBDT.load_from_json(js2, 'deltaboost')
            cr = compare_tree(gbdt.trees[0], gbdt_remain.trees[0])
            print(f"{np.count_nonzero([0 if node.is_leaf else 1 for node in gbdt.trees[0].nodes])}\t"
                  f"{np.count_nonzero([0 if node.is_leaf else 1 for node in gbdt_remain.trees[0].nodes])}")
            for level in [4, 6, 8, 10]:
                # print(f"{level=}: {cr.n_common_node(level)=}, {cr.n_node_common_feature(level)=}, "
                #       f"{cr.n_node_split_nbr_include(level)=}, {cr.n_node_common_split_bid(level)=}")
                print(f"{cr.n_common_node(level)}\t{cr.n_node_common_feature(level)}\t"
                      f"{cr.n_node_split_nbr_include(level)}\t{cr.n_node_common_split_bid(level)}")
        print(f"Alpha {alpha:.1f} done.")
