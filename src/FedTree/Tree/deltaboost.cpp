//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/Tree/deltaboost.h"
#include "FedTree/booster.h"
#include "FedTree/deltabooster.h"

void DeltaBoost::train(GBDTParam &param, DataSet &dataset) {
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist") {
        std::cout << "FedTree only supports histogram-based training yet";
        exit(1);
    }

    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if (param.num_class > 2)
            param.tree_per_rounds = param.num_class;
    } else if (param.objective.find("reg:") != std::string::npos) {
        param.num_class = 1;
    }

//    std::map<int, vector<int>> batch_idxs;
//    Partition partition;
//    vector<DataSet> subsets(3);
//    partition.homo_partition(dataset, 3, true, subsets, batch_idxs);
//

    DeltaBooster booster;
    booster.init(dataset, *(reinterpret_cast<DeltaBoostParam*>(&param)));   // todo: too ugly and unsafe, fix it
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    for (int i = 0; i < param.n_trees; ++i) {
        //one iteration may produce multiple trees, depending on objectives
        booster.boost(trees);
    }

//    float_type score = predict_score(param, dataset);
//    LOG(INFO) << score;

    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count();
    return;
}

float_type DeltaBoost::predict_score(const GBDTParam &model_param, const DataSet &dataset) {
    return GBDT::predict_score(model_param, dataset);
    // todo: implement prediction
}

void DeltaBoost::predict_raw(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict) {
    TIMED_SCOPE(timerObj, "predict");
    int n_instances = dataSet.n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = trees.size();
    int num_class = trees.front().size();
    int num_node = trees[0][0].nodes.size();

    int total_num_node = num_iter * num_class * num_node;
    //TODO: reduce the output size for binary classification
    y_predict.resize(n_instances * num_class);

    SyncArray<DeltaTree::DeltaNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:trees) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(DeltaTree::DeltaNode) * num_node);
            tree_cnt++;
        }
    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");
    //copy instances from to GPU
    SyncArray<int> csr_col_idx(dataSet.csr_col_idx.size());
    SyncArray<float_type> csr_val(dataSet.csr_val.size());
    SyncArray<int> csr_row_ptr(dataSet.csr_row_ptr.size());
    csr_col_idx.copy_from(dataSet.csr_col_idx.data(), dataSet.csr_col_idx.size());
    csr_val.copy_from(dataSet.csr_val.data(), dataSet.csr_val.size());
    csr_row_ptr.copy_from(dataSet.csr_row_ptr.data(), dataSet.csr_row_ptr.size());

    //do prediction
    auto model_host_data = model.host_data();
    auto predict_data = y_predict.host_data();
    auto csr_col_idx_data = csr_col_idx.host_data();
    auto csr_val_data = csr_val.host_data();
    auto csr_row_ptr_data = csr_row_ptr.host_data();
    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

    //predict BLOCK_SIZE instances in a block, 1 thread for 1 instance
//    int BLOCK_SIZE = 128;
    //determine whether we can use shared memory
//    size_t smem_size = n_features * BLOCK_SIZE * sizeof(float_type);
//    int NUM_BLOCK = (n_instances - 1) / BLOCK_SIZE + 1;

    //use sparse format and binary search
#pragma omp parallel for
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](DeltaTree::DeltaNode node, float_type feaValue) {
            return feaValue < node.split_value ? node.lch_index : node.rch_index;
        };
        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };
        int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                const DeltaTree::DeltaNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
                DeltaTree::DeltaNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int fid = curNode.split_feature_id;
                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    if (!is_missing)
                        cur_nid = get_next_child(curNode, fval);
                    else if (curNode.default_right)
                        cur_nid = curNode.rch_index;
                    else
                        cur_nid = curNode.lch_index;
                    curNode = node_data[cur_nid];
                }
                sum += lr * node_data[cur_nid].base_weight;
            }
            predict_data_class[iid] += sum;
        }//end all tree prediction
    }
}
