//
// Created by HUSTW on 8/17/2021.
//

#include <utility>

#include "FedTree/Tree/tree.h"

#ifndef FEDTREE_DELTA_TREE_REMOVER_H
#define FEDTREE_DELTA_TREE_REMOVER_H

class DeltaTreeRemover {
public:
    DeltaTreeRemover() = default;

    DeltaTreeRemover(const DeltaTree& tree, DeltaBoostParam param, vector<GHPair> gh_pairs):
    tree(tree), param(std::move(param)), gh_pairs(std::move(gh_pairs)) { }

    bool remove_sample_by_id(int id);

    DeltaTree tree;
    DeltaBoostParam param;
    vector<GHPair> gh_pairs;

};

#endif //FEDTREE_DELTA_TREE_REMOVER_H
