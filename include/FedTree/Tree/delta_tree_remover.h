//
// Created by HUSTW on 8/17/2021.
//

#include <utility>
#include <FedTree/dataset.h>

#include "FedTree/Tree/tree.h"

#ifndef FEDTREE_DELTA_TREE_REMOVER_H
#define FEDTREE_DELTA_TREE_REMOVER_H

class DeltaTreeRemover {
public:
    DeltaTreeRemover() = default;

    DeltaTreeRemover(DeltaTree* tree_ptr, const DataSet* dataSet, DeltaBoostParam param, vector<GHPair> gh_pairs):
    tree_ptr(tree_ptr), dataSet(dataSet), param(std::move(param)), gh_pairs(std::move(gh_pairs)) { }

    vector<int> remove_sample_by_id(int id);
    vector<int> adjust_gradients_by_indices(const vector<int>& indices, const vector<GHPair>& delta_gh_pairs);

    DeltaTree *tree_ptr;
    DeltaBoostParam param;
    vector<GHPair> gh_pairs;

    const DataSet* dataSet;
};

#endif //FEDTREE_DELTA_TREE_REMOVER_H
