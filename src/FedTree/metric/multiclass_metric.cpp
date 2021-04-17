//
// Created by liqinbin on 12/11/20.
//

#include "FedTree/metric/multiclass_metric.h"
#include "FedTree/util/device_lambda.h"
#include "thrust/reduce.h"
#include "thrust/execution_policy.h"


float_type MulticlassAccuracy::get_score(const SyncArray<float_type> &y_p) const {
    CHECK_EQ(num_class * y.size(), y_p.size()) << num_class << " * " << y.size() << " != " << y_p.size();
    int n_instances = y.size();
    auto y_data = y.host_data();
    auto yp_data = y_p.host_data();
    SyncArray<int> is_true(n_instances);
    auto is_true_data = is_true.host_data();
    int num_class = this->num_class;
#pragma omp parallel for
    for (int i = 0; i < n_instances; i++){
        int max_k = 0;
        float_type max_p = yp_data[i];
        for (int k = 1; k < num_class; ++k) {
            if (yp_data[k * n_instances + i] > max_p) {
                max_p = yp_data[k * n_instances + i];
                max_k = k;
            }
        }
        is_true_data[i] = max_k == y_data[i];
    }

    float acc = thrust::reduce(thrust::host, is_true_data, is_true_data + n_instances) / (float) n_instances;
    return acc;
}

float_type BinaryClassMetric::get_score(const SyncArray<float_type> &y_p) const {
    int n_instances = y.size();
    auto y_data = y.host_data();
    auto yp_data = y_p.host_data();
    SyncArray<int> is_true(n_instances);
    auto is_true_data = is_true.host_data();
#pragma omp parallel for
    for (int i = 0; i < n_instances; i++){
        // change the threshold to 0 if the classes are -1 and 1 and using regression as the objective.
        int max_k = (yp_data[i] > 0.5) ? 1 : 0;
        is_true_data[i] = max_k == y_data[i];
    }
    float acc = thrust::reduce(thrust::host, is_true_data, is_true_data + n_instances) / (float) n_instances;
    return 1 - acc;
}
