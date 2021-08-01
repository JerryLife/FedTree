//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_GBDT_H
#define FEDTREE_GBDT_H

//Todo: the GBDT model, train a tree, update gradients
#include "tree.h"
#include "FedTree/dataset.h"

class GBDT {
public:
    vector<vector<Tree>> trees;

    GBDT() = default;

    explicit GBDT(const vector<vector<Tree>>& gbdt){
        trees = gbdt;
    }

    virtual void train(GBDTParam &param, DataSet &dataset);

    virtual vector<float_type> predict(const GBDTParam &model_param, const DataSet &dataSet);

    virtual void predict_raw(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict);

    virtual void predict_raw_vertical(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict, std::map<int, vector<int>> &batch_idxs);

    virtual float_type predict_score(const GBDTParam &model_param, const DataSet &dataSet);

    virtual float_type predict_score_vertical(const GBDTParam &model_param, const DataSet &dataSet, std::map<int, vector<int>> &batch_idxs);
};

#endif //FEDTREE_GBDT_H
