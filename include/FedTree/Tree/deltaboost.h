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

    void train(GBDTParam &param, DataSet &dataset) override;

    float_type predict_score(const GBDTParam &model_param, const DataSet &dataSet) override;

    void predict_raw(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict) override;
};

#endif //FEDTREE_DELTABOOST_H
