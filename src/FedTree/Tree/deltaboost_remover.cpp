#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
//
// Created by zhaomin on 16/12/21.
//

#include "FedTree/Tree/deltaboost_remover.h"

void DeltaBoostRemover::get_info_by_prediction() {

    auto& trees = *trees_ptr;
    size_t n_instances = dataSet->n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    size_t num_iter = param.n_trees == -1 ? trees.size() : param.n_used_trees;
    int num_class = static_cast<int>(trees.front().size());
    std::vector<std::vector<float_type>> y_predict(n_instances, std::vector<float_type>(num_class, 0));

    //copy instances from to GPU
    SyncArray<int> csr_col_idx(dataSet->csr_col_idx.size());
    SyncArray<float_type> csr_val(dataSet->csr_val.size());
    SyncArray<int> csr_row_ptr(dataSet->csr_row_ptr.size());
    csr_col_idx.copy_from(dataSet->csr_col_idx.data(), dataSet->csr_col_idx.size());
    csr_val.copy_from(dataSet->csr_val.data(), dataSet->csr_val.size());
    csr_row_ptr.copy_from(dataSet->csr_row_ptr.data(), dataSet->csr_row_ptr.size());

    //do prediction
    auto csr_col_idx_data = csr_col_idx.host_data();
    auto csr_val_data = csr_val.host_data();
    auto csr_row_ptr_data = csr_row_ptr.host_data();
    auto lr = param.learning_rate;

    auto get_val = [](const int *row_idx, const float_type *row_val, int row_len, int idx,
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

    auto get_next_child = [](const DeltaTree::DeltaNode& node, float_type feaValue) {
        return feaValue < node.split_value ? node.lch_index : node.rch_index;
    };

    //use sparse format and binary search
#pragma omp parallel for default(none) shared(n_instances, csr_col_idx_data, csr_row_ptr_data, csr_val_data, num_class, \
                                          num_iter, trees, get_val, lr, y_predict, get_next_child)    // remove for debug
    for (int iid = 0; iid < n_instances; ++iid) {
        int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                SyncArray<float_type> y_arr(1); y_arr.host_data()[0] = dataSet->y[iid];
                SyncArray<float_type> y_pred_arr(1); y_pred_arr.host_data()[0] = sum;
                SyncArray<GHPair> gh_pair_arr(1);
                obj->get_gradient(y_arr, y_pred_arr, gh_pair_arr);
                tree_removers[iter].gh_pairs[iid] = gh_pair_arr.host_data()[0];

                const DeltaTree::DeltaNode *end_leaf;
                const auto &nodes = trees[iter][t].nodes;
                std::vector<int> visiting_node_indices = {0};
                std::vector<bool> prior_flags = {true};
                while (!visiting_node_indices.empty()) {        // DFS
                    int node_id = visiting_node_indices.back();
                    visiting_node_indices.pop_back();
                    bool is_prior = prior_flags.back();
                    prior_flags.pop_back();
                    const auto& node = nodes[node_id];
                    int fid = node.split_feature_id;
                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);

                    tree_removers[iter].ins2node_indices[iid].push_back(node_id);

                    if (node.is_leaf) {
                        if (is_prior) {
                            end_leaf = &node;
                        }
                    } else {
                        // potential nodes (if any)
                        if (is_prior){
                            for (int j = 1; j < node.potential_nodes_indices.size(); ++j) {
                                visiting_node_indices.push_back(node.potential_nodes_indices[j]);
                                prior_flags.push_back(false);
                            }
                        }

                        // prior node
                        if (!is_missing) {
                            int child_id = get_next_child(nodes[node_id], fval);
                            visiting_node_indices.push_back(child_id);
                        } else if (node.default_right) {
                            visiting_node_indices.push_back(node.rch_index);
                        } else {
                            visiting_node_indices.push_back(node.lch_index);
                        }
                        prior_flags.push_back(is_prior);
                    }
                }
                sum += lr * end_leaf->base_weight;
            }
            y_predict[iid][t] += sum;
        }
    }

}
