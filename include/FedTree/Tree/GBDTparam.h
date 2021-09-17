//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_GBDTPARAM_H
#define FEDTREE_GBDTPARAM_H

#include <string>
#include <FedTree/common.h>

// Todo: gbdt params, refer to ThunderGBM https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/common.h
struct GBDTParam {
    int depth;
    int n_trees;
    float_type min_child_weight;
    float_type lambda;
    float_type gamma;
    float_type rt_eps;
    float column_sampling_rate;
    std::string path;
    std::string test_path;
    int verbose;
    bool profiling;
    bool bagging;
    int n_parallel_trees;
    float learning_rate;
    std::string objective;
    int num_class;
    int tree_per_rounds; // #tree of each round, depends on #class

    //for histogram
    int max_num_bin;

    int n_device;

    std::string tree_method;
    std::string metric;

};

struct DeltaBoostParam : public GBDTParam {
    bool enable_delta = false;
    float_type remove_ratio = 0.0;

    DeltaBoostParam() = default;

    DeltaBoostParam(const GBDTParam & gbdt_param, bool enable_delta, float_type remove_ratio):
    GBDTParam(gbdt_param), enable_delta(enable_delta), remove_ratio(remove_ratio) {}
};

#endif //FEDTREE_GBDTPARAM_H
