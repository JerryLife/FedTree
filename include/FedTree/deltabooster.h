//
// Created by HUSTW on 7/31/2021.
//

#include "booster.h"
#include "Tree/GBDTparam.h"
#include "FedTree/Tree/delta_tree_builder.h"

#ifndef FEDTREE_DELTABOOSTER_H
#define FEDTREE_DELTABOOSTER_H

class DeltaBooster : public Booster {
public:
    void init(DataSet &dataSet, const DeltaBoostParam &param, bool get_cut_points = true);

//    void init(const GBDTParam &param, int n_instances) override;
    void boost(vector<vector<DeltaTree>>& boosted_model, vector<vector<GHPair>>& gh_pairs_per_sample,
               vector<vector<vector<int>>>& ins2node_indices_per_tree);
    void boost_without_prediction(vector<vector<DeltaTree>>& boosted_model);

    std::unique_ptr<DeltaTreeBuilder> fbuilder;
    DeltaBoostParam param;
};

#endif //FEDTREE_DELTABOOSTER_H
