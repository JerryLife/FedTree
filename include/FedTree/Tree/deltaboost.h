//
// Created by HUSTW on 7/31/2021.
//

#include "gbdt.h"

#ifndef FEDTREE_DELTABOOST_H
#define FEDTREE_DELTABOOST_H

class DeltaBoost : public GBDT{
public:
    vector<vector<DeltaTree>> trees;

    vector<vector<GHPair>> gh_pairs_per_sample;       // first index is the iteration, second index is the sample ID


    DeltaBoost() = default;

    explicit DeltaBoost(const vector<vector<DeltaTree>>& gbdt){
        trees = gbdt;
    }

    void train(DeltaBoostParam &param, DataSet &dataset);

    float_type predict_score(const DeltaBoostParam &model_param, const DataSet &dataSet, int num_trees=-1);

    void predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict,
                                 int num_trees=-1);

    void remove_samples(DeltaBoostParam &param, DataSet &dataset, const vector<int>& sample_indices);
};

#endif //FEDTREE_DELTABOOST_H
