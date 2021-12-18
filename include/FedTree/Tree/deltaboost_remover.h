//
// Created by zhaomin on 16/12/21.
//

#ifndef FEDTREE_DELTABOOST_REMOVER_H
#define FEDTREE_DELTABOOST_REMOVER_H


#include <FedTree/dataset.h>
#include <FedTree/objective/objective_function.h>

#include "FedTree/Tree/tree.h"
#include "FedTree/Tree/delta_tree_remover.h"
#include "deltaboost.h"
#include <FedTree/metric/metric.h>

class DeltaBoostRemover {
public:

    DeltaBoostRemover(const DataSet *dataSet, DeltaBoost *deltaboostPtr, ObjectiveFunction *obj,
                      const DeltaBoostParam &param) :
    dataSet(dataSet), deltaboost_ptr(deltaboostPtr), obj(obj), param(param) {
        for (int i = 0; i < param.n_used_trees; ++i) {
            tree_removers.emplace_back(*(std::unique_ptr<DeltaTreeRemover>(
                    new DeltaTreeRemover(&(deltaboostPtr->trees[i][0]), dataSet, param))));
        }
    }

    void get_info_by_prediction();      // get initial info in each deltatree remover

    std::vector<DeltaTreeRemover> tree_removers;

private:
    const DataSet* dataSet = nullptr;
    DeltaBoost* deltaboost_ptr = nullptr;
    ObjectiveFunction *obj = nullptr;
    DeltaBoostParam param;
};


#endif //FEDTREE_DELTABOOST_REMOVER_H
