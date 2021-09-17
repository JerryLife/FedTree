//
// Created by HUSTW on 7/31/2021.
//

#include "gbdt.h"

#ifndef FEDTREE_DELTABOOST_H
#define FEDTREE_DELTABOOST_H

class DeltaBoost : public GBDT{
public:
    vector<vector<DeltaTree>> trees;

    DeltaBoost() = default;

    explicit DeltaBoost(const vector<vector<DeltaTree>>& gbdt){
        trees = gbdt;
    }

    void train(DeltaBoostParam &param, DataSet &dataset);

    float_type predict_score(const DeltaBoostParam &model_param, const DataSet &dataSet);

    void predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict);

    void remove_samples(const vector<int>& sample_indices);
};

#endif //FEDTREE_DELTABOOST_H
