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

#include <memory>

class DeltaBoostRemover {
public:
    DeltaBoostRemover() = default;

    DeltaBoostRemover(const DataSet *dataSet, std::vector<std::vector<DeltaTree>>* trees_ptr,
                      const vector<vector<bool>> &is_subset_indices_in_trees,
                      ObjectiveFunction *obj, const DeltaBoostParam &param) :
    dataSet(dataSet), trees_ptr(trees_ptr),  obj(obj), param(param) {
        for (int i = 0; i < param.n_used_trees; ++i) {
            tree_removers.emplace_back(DeltaTreeRemover(
                    &((*trees_ptr)[i][0]), dataSet, param, is_subset_indices_in_trees[i]));
        }
    }

    DeltaBoostRemover(const DataSet *dataSet, std::vector<std::vector<DeltaTree>>* trees_ptr,
                      ObjectiveFunction *obj, const DeltaBoostParam &param) {

        typedef std::chrono::high_resolution_clock clock;

        auto start_time_all = clock::now();

        auto start_time = clock::now();
        this->dataSet = dataSet;
        auto end_time = clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        LOG(DEBUG) << "[Removing time] Step 0.1 = " << duration.count();

        start_time = clock::now();
        this->trees_ptr = trees_ptr;
        end_time = clock::now();
        duration = end_time - start_time;
        LOG(DEBUG) << "[Removing time] Step 0.2 = " << duration.count();

        start_time = clock::now();
        this->obj = obj;
        end_time = clock::now();
        duration = end_time - start_time;
        LOG(DEBUG) << "[Removing time] Step 0.3 = " << duration.count();

        start_time = clock::now();
        this->param = param;
        end_time = clock::now();
        duration = end_time - start_time;
        LOG(DEBUG) << "[Removing time] Step 0.4 = " << duration.count();

        start_time = clock::now();
        for (int i = 0; i < param.n_used_trees; ++i) {
            tree_removers.emplace_back(DeltaTreeRemover(
                    &((*trees_ptr)[i][0]), dataSet, param));
        }
        end_time = clock::now();
        duration = end_time - start_time;
        LOG(DEBUG) << "[Removing time] Step 0.5 = " << duration.count();

        end_time = clock::now();
        duration = end_time - start_time_all;
        LOG(DEBUG) << "[Removing time] Step 0 (in) = " << duration.count();
    }

    void get_info_by_prediction();      // get initial info in each deltatree remover

    std::vector<DeltaTreeRemover> tree_removers;

private:
    const DataSet* dataSet = nullptr;
    std::vector<std::vector<DeltaTree>>* trees_ptr = nullptr;
    ObjectiveFunction *obj = nullptr;
    DeltaBoostParam param;
};


#endif //FEDTREE_DELTABOOST_REMOVER_H
