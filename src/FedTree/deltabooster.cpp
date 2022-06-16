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
                         vector<vector<vector<int>>>& ins2node_indices_per_tree) {
    TIMED_FUNC(timerObj);
//    std::unique_lock<std::mutex> lock(mtx);

    //update gradients
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);
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


