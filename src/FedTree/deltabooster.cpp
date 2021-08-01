//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/deltabooster.h"

void DeltaBooster::init(DataSet &dataSet, const DeltaBoostParam &delta_param, bool get_cut_points) {
    param = delta_param;

    fbuilder.reset(new DeltaTreeBuilder);
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

//void DeltaBooster::init(const GBDTParam &param, int n_instances) {
//
//}

void DeltaBooster::boost(vector<vector<DeltaTree>> &boosted_model) {
    TIMED_FUNC(timerObj);
//    std::unique_lock<std::mutex> lock(mtx);
    //update gradients
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);

//    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_delta_approximate(gradients));

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    std::ofstream myfile;
    myfile.open ("data.txt", std::ios_base::app);
    myfile << metric->get_score(fbuilder->get_y_predict()) << "\n";
    myfile.close();
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}
