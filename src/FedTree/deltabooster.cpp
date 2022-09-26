//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/deltabooster.h"

#include <memory>
#include "FedTree/Tree/deltaboost.h"


void DeltaBooster::init(DataSet &dataSet, const DeltaBoostParam &delta_param, bool get_cut_points) {
    param = delta_param;

    fbuilder = std::make_unique<DeltaTreeBuilder>();
    if(get_cut_points)
        fbuilder->init(dataSet, param);
    else {
        fbuilder->init_nocutpoints(dataSet, param);
    }
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    if (param.metric == "default")
        metric.reset(Metric::create(obj->default_metric_name()));
    else
        metric.reset(Metric::create(param.metric));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = SyncArray<GHPair>(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}


void DeltaBooster::reset(DataSet &dataSet, const DeltaBoostParam &delta_param, bool get_cut_points) {
    param = delta_param;

//    fbuilder = std::make_unique<DeltaTreeBuilder>();
    if(get_cut_points)
        fbuilder->reset(dataSet, param);
    else {
        LOG(FATAL) << "Not supported yet";
        fbuilder->init_nocutpoints(dataSet, param);
    }
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    if (param.metric == "default")
        metric.reset(Metric::create(obj->default_metric_name()));
    else
        metric.reset(Metric::create(param.metric));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = SyncArray<GHPair>(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}


void DeltaBooster::boost(vector<vector<DeltaTree>>& boosted_model, vector<vector<GHPair>>& gh_pairs_per_sample,
                         vector<vector<vector<int>>>& ins2node_indices_per_tree, const vector<int> &row_hash) {
    TIMED_FUNC(timerObj);
//    std::unique_lock<std::mutex> lock(mtx);

    //update gradients
    SyncArray<GHPair> original_gh(gradients.size());
    obj->get_gradient(y, fbuilder->get_y_predict(), original_gh);

    // quantize gradients if needed. todo: optimize these per-instance copy
    if (param.n_quantize_bins > 0) {
        gradients.load_from_vec(quantize_gradients(original_gh.to_vec(), param.n_quantize_bins, row_hash));
    } else {
        gradients.copy_from(original_gh);
    }

    gh_pairs_per_sample.push_back(gradients.to_vec());
    std::vector<std::vector<int>> ins2node_indices;
//    if (param.bagging) rowSampler.do_bagging(gradients);

    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_delta_approximate(gradients, ins2node_indices));
    PERFORMANCE_CHECKPOINT(timerObj);
    ins2node_indices_per_tree.push_back(ins2node_indices);

    //show metric on training set
    std::ofstream myfile;
    myfile.open ("data.txt", std::ios_base::app);
    myfile << fbuilder->get_y_predict() << "\n";
    myfile.close();
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}

vector<GHPair> DeltaBooster::quantize_gradients(const vector<GHPair> &gh, int n_bins, const vector<int> &row_hash) {
    /**
     * Randomly quantize gradients and hessians to neighboring grids.
     */
    vector<GHPair> quantized_gh(gh.size());

    // get max absolute value of gh.g and gh.h
    float_type max_abs_g = 0, max_abs_h = 0;
    for (int i = 0; i < gh.size(); ++i) {
        max_abs_g = std::max(max_abs_g, std::abs(gh[i].g));
        max_abs_h = std::max(max_abs_h, std::abs(gh[i].h));
    }

    // calculate bin width
    float_type bin_width_g = max_abs_g / n_bins;
    float_type bin_width_h = max_abs_h / (n_bins * 2);      // smaller width according to the NeurIPS-22 paper

//    std::mt19937 gen1(seed);
//    std::mt19937 gen2(seed + 1);

    // random round gh to integers (DO NOT run in parallel to ensure random sequence is the same)
    for (int i = 0; i < gh.size(); ++i) {
        quantized_gh[i].g = random_round(gh[i].g / bin_width_g, row_hash[i]) * bin_width_g;
        quantized_gh[i].h = random_round(gh[i].h / bin_width_h, row_hash[i] + 1) * bin_width_h;
    }

    auto sum_gh = std::accumulate(quantized_gh.begin(), quantized_gh.end(), GHPair(0, 0));

    return quantized_gh;
}

float_type DeltaBooster::random_round(float_type x, float_type left, float_type right, size_t seed) {
    /*
     * Randomly round x to the left or right. The expected value of the result is x.
     * The probability of rounding to the left is (right - x) / (right - left);
     * The probability of rounding to the right is (x - left) / (right - left);
     */
    float_type prob = (x - left) / (right - left);
    if (prob < 0 || prob > 1) {
        LOG(FATAL) << "prob = " << prob << ", left = " << left << ", right = " << right << ", x = " << x;
    }
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0, 1);
    float_type rand = dis(gen);
    if (rand > prob) {
        return std::floor(x);
    } else {
        return std::ceil(x);
    }
}

float_type DeltaBooster::random_round(float_type x, size_t seed) {
    /*
     * Randomly round x to the floor or ceiling integer (of float_type). The expected value of the result is x.
     * The probability of rounding to the ceiling is (x - floor(x));
     * The probability of rounding to the floor is (ceiling(x) - x);
     */

    std::mt19937 gen{seed};
    std::uniform_real_distribution<> dis(0, 1);
    float_type rand = dis(gen);
    if (rand > x - std::floor(x)) {
        return std::floor(x);
    } else {
        return std::ceil(x);
    }
}






