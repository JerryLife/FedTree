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

    std::string remain_data_path;
    std::string delete_data_path;
    std::string save_model_name;

};

struct DeltaBoostParam : public GBDTParam {
    bool enable_delta = false;
    std::string dataset_name;
    float_type remove_ratio = 0.0;
    float_type min_diff_gain = 0;
    float_type max_range_gain = 0;
    int n_used_trees = 0;


    DeltaBoostParam() = default;

    DeltaBoostParam(const GBDTParam *gbdt_param, const DeltaBoostParam *deltaboost_param):
    GBDTParam(*gbdt_param), enable_delta(deltaboost_param->enable_delta),
    remove_ratio(deltaboost_param->remove_ratio),
    min_diff_gain(deltaboost_param->min_diff_gain),
    max_range_gain(deltaboost_param->max_range_gain),
    dataset_name(deltaboost_param->dataset_name),
    n_used_trees(deltaboost_param->n_used_trees) {
        if (deltaboost_param->n_used_trees > 0) {
            this->n_used_trees = deltaboost_param->n_used_trees;
        } else {
            this->n_used_trees = gbdt_param->n_trees;
        }
    }
};

#endif //FEDTREE_GBDTPARAM_H
