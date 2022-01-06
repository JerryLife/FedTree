from typing import List
import abc

import numpy as np


class Evaluative(abc.ABC):
    @abc.abstractmethod
    def predict_score(self, X):
        pass


class Node:
    def __init__(self, is_leaf=False, split_feature_id=None, split_value=None, base_weight=None, default_right=False,
                 lch_id=-1, rch_id=-1, parent_id=-1):
        self.parent_id = parent_id
        self.lch_id = lch_id
        self.rch_id = rch_id
        self.split_feature_id = split_feature_id
        self.split_value = split_value
        self.base_weight = base_weight
        self.default_right = default_right
        self.is_leaf = is_leaf

    def decide_right(self, x) -> bool:
        """
        :param x: data for prediction
        :return: false for left, true for right
        """
        feature_value = x[self.split_feature_id]
        if abs(feature_value) < 1e-7:
            # missing value
            return self.default_right
        return feature_value > self.split_value

    @classmethod
    def load_from_json(cls, js: dict):
        return cls(parent_id=int(js['parent_index']),
                   lch_id=int(js['lch_index']),
                   rch_id=int(js['rch_index']),
                   split_feature_id=int(js['split_feature_id']),
                   split_value=float(js['split_value']),
                   base_weight=float(js['base_weight']),
                   default_right=bool(js['default_right']),
                   is_leaf=bool(js['is_leaf']))


class Tree:
    def __init__(self, nodes: List[Node] = None):
        self.nodes = nodes if nodes is not None else []

    def add_child_(self, node: Node, is_right):
        self.nodes.append(node)
        if is_right:
            self.nodes[node.parent_id].rch_id = len(self.nodes) - 1
        else:
            self.nodes[node.parent_id].lch_id = len(self.nodes) - 1

    def add_root_(self, node: Node):
        assert self.nodes is None or len(self.nodes) == 0
        self.nodes = [node]

    def predict(self, x):
        node = self.nodes[0]
        while not node.is_leaf:
            if node.decide_right(x):
                node = self.nodes[node.rch_id]
            else:
                node = self.nodes[node.lch_id]
        return node.base_weight

    @classmethod
    def load_from_json(cls, js: dict):
        tree = cls()
        visiting_node_indices = [0]
        while len(visiting_node_indices) > 0:
            node_id = visiting_node_indices.pop(0)
            node = Node.load_from_json(js['nodes'][node_id])
            is_right_child = int(tree.nodes[node.parent_id].rch_id) == node_id if node.parent_id != -1 else False

            if not node.is_leaf:
                js['nodes'][node.lch_id]['parent_index'] = len(tree.nodes)
                js['nodes'][node.rch_id]['parent_index'] = len(tree.nodes)
                visiting_node_indices += [node.lch_id, node.rch_id]

            tree.add_child_(node, is_right_child)
        return tree


class GBDT(Evaluative):
    def __init__(self, lr=1., trees=None):
        self.lr = lr
        self.trees = trees

    def predict_score(self, X: np.ndarray):
        """
        :param X: 2D array
        :return: y: 1D array
        """
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            score = 0
            for tree in self.trees:
                score += tree.predict(x) * self.lr
            scores[i] = score
        return scores

    def predict(self, X: np.ndarray, task='bin-cls'):
        if task == 'bin-cls':
            return np.where(self.predict_score(X) > 0.5, 1, 0)
        else:
            assert False, "Task not supported."

    @classmethod
    def load_from_json(cls, js, type='deltaboost'):
        assert type in ['deltaboost', 'gbdt'], "Unsupported type"

        gbdt = cls(lr=js['learning_rate'], trees=[])
        for tree_js in js[type]['trees']:
            gbdt.trees.append(Tree.load_from_json(tree_js[0]))
        return gbdt

