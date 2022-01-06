//
// Created by HUSTW on 8/17/2021.
//

#include <algorithm>
#include <queue>
#include <random>
#include <chrono>

#include "FedTree/Tree/delta_tree_remover.h"

//bool DeltaTreeRemover::remove_samples_by_indices(const vector<int>& indices) {
//    /**
//     * @param id: the index of sample to be removed from the tree
//     * @return : true when a removal is successful; false when failing to remove and a retrain is needed
//     */
//
//    for (int id: indices) {
//        remove_sample_by_id(id);
//    }
//}

void DeltaTreeRemover::remove_sample_by_id(int id) {
    vector<int> indices = {id};
    vector<GHPair> gh_pair_vec = {-gh_pairs[id]};
    adjust_gradients_by_indices(indices, gh_pair_vec);
}

void DeltaTreeRemover::remove_samples_by_indices(const vector<int>& indices) {
    vector<GHPair> gh_pair_vec(indices.size());

#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
        gh_pair_vec[i] = -gh_pairs[indices[i]];
    }

    // this function is parallel
    adjust_gradients_by_indices(indices, gh_pair_vec);
}




void DeltaTreeRemover::adjust_gradients_by_indices(const vector<int>& indices, const vector<GHPair>& delta_gh_pairs) {
    /**
     * @param id: the indices of sample to be adjusted gradients
     * @param delta_gh_pair: gradient and hessian to be subtracted from the tree
     * @return : indices of all modified leaves
     */

    SyncArray<int> csr_col_idx(dataSet->csr_col_idx.size());
    SyncArray<float_type> csr_val(dataSet->csr_val.size());
    SyncArray<int> csr_row_ptr(dataSet->csr_row_ptr.size());
    csr_col_idx.copy_from(dataSet->csr_col_idx.data(), dataSet->csr_col_idx.size());
    csr_val.copy_from(dataSet->csr_val.data(), dataSet->csr_val.size());
    csr_row_ptr.copy_from(dataSet->csr_row_ptr.data(), dataSet->csr_row_ptr.size());

    const auto csr_col_idx_data = csr_col_idx.host_data();
    const auto csr_val_data = csr_val.host_data();
    const auto csr_row_ptr_data = csr_row_ptr.host_data();

    // update the gain of all nodes according to ins2node_indices
    vector<vector<int>> updating_node_indices(indices.size(), vector<int>(0));
#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
        updating_node_indices[i] = ins2node_indices[indices[i]];
    }

    auto get_val = [&](int iid, int fid,
                   bool *is_missing) -> float_type {
        int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];

        //binary search to get feature value
        const int *left = col_idx;
        const int *right = col_idx + row_len;

        while (left != right) {
            const int *mid = left + (right - left) / 2;
            if (*mid == fid) {
                *is_missing = false;
                return row_val[mid - col_idx];
            }
            if (*mid > fid)
                right = mid;
            else left = mid + 1;
        }
        *is_missing = true;
        return 0;
    };

    // update GH_pair of node and parent (parallel)
#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            // update sum_gh_pair
            auto &node = tree_ptr->nodes[node_id];
            #pragma omp atomic
            node.sum_gh_pair.g += delta_gh_pairs[i].g;
            #pragma omp atomic
            node.sum_gh_pair.h += delta_gh_pairs[i].h;
            #pragma omp atomic
            node.gain.self_g += delta_gh_pairs[i].g;
            #pragma omp atomic
            node.gain.self_h += delta_gh_pairs[i].h;

            // update missing_gh
            bool is_missing;
            float_type split_fval = get_val(indices[i], node.split_feature_id, &is_missing);
            if (is_missing) {
                #pragma omp atomic
                node.gain.missing_g += delta_gh_pairs[i].g;
                #pragma omp atomic
                node.gain.missing_h += delta_gh_pairs[i].h;
            }
        }
    }


#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            auto &node = tree_ptr->nodes[node_id];
            if (!node.is_leaf) {
                node.gain.lch_g = tree_ptr->nodes[node.lch_index].gain.self_g;
                node.gain.lch_h = tree_ptr->nodes[node.lch_index].gain.self_h;
                node.gain.rch_g = tree_ptr->nodes[node.rch_index].gain.self_g;
                node.gain.rch_h = tree_ptr->nodes[node.rch_index].gain.self_h;
            }
        }
    }

    // recalculate direction
#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            auto &node = tree_ptr->nodes[node_id];

            node.calc_weight(param.lambda);     // this lambda should be consistent with the training

            if (!node.is_leaf) {
//                node.gain.gain_value = node.default_right ? -node.gain.cal_gain_value() : node.gain.cal_gain_value();     // calculate original gain value

                // recalculate default direction
                if (node.default_right) {
                    node.gain.gain_value = -node.gain.cal_gain_value();
                    assert(node.gain.gain_value <= 0);
                    DeltaTree::DeltaGain default_left_gain(node.gain);
#pragma omp atomic
                    default_left_gain.lch_g += node.gain.missing_g;
#pragma omp atomic
                    default_left_gain.lch_h += node.gain.missing_h;
#pragma omp atomic
                    default_left_gain.rch_g -= node.gain.missing_g;
#pragma omp atomic
                    default_left_gain.rch_h -= node.gain.missing_h;
                    default_left_gain.gain_value = default_left_gain.cal_gain_value();
                    if (fabs(default_left_gain.gain_value) > fabs(node.gain.gain_value)) {
                        // switch default direction
                        node.gain = default_left_gain;
                        node.default_right = false;
                    }
                } else {
                    node.gain.gain_value = node.gain.cal_gain_value();
                    assert(node.gain.gain_value >= 0);
                    DeltaTree::DeltaGain default_right_gain(node.gain);
#pragma omp atomic
                    default_right_gain.rch_g += node.gain.missing_g;
#pragma omp atomic
                    default_right_gain.rch_h += node.gain.missing_h;
#pragma omp atomic
                    default_right_gain.lch_g -= node.gain.missing_g;
#pragma omp atomic
                    default_right_gain.lch_h -= node.gain.missing_h;
                    default_right_gain.gain_value = -default_right_gain.cal_gain_value();
                    if (fabs(default_right_gain.gain_value) > fabs(node.gain.gain_value)) {
                        // switch default direction
                        default_right_gain.gain_value = -default_right_gain.gain_value;
                        node.gain = default_right_gain;
                        node.default_right = true;
                    }
                }
            }
        }
    }

    sort_potential_nodes_by_gain(0);
}

void DeltaTreeRemover::sort_potential_nodes_by_gain(int root_idx) {
    std::queue<int> processing_nodes;
    processing_nodes.push(root_idx);    // start from root node
    while(!processing_nodes.empty()) {
        int nid = processing_nodes.front();

        processing_nodes.pop();
        auto& node = tree_ptr->nodes[nid];

        if (node.is_leaf) {
            continue;
        }

        if (!node.is_valid) {
            continue;
        }

        if (!node.is_robust()) {
            // sort the nodes by descending order of gain
            std::sort(node.potential_nodes_indices.begin(), node.potential_nodes_indices.end(),
                      [&](int i, int j){
                          return fabs(tree_ptr->nodes[i].gain.gain_value) > fabs(tree_ptr->nodes[j].gain.gain_value);
                      });

            // sync the order through potential nodes
            for (int j: node.potential_nodes_indices) {
                auto &potential_node = tree_ptr->nodes[j];
                potential_node.potential_nodes_indices = node.potential_nodes_indices;
                if (!potential_node.is_leaf) {
                    processing_nodes.push(potential_node.lch_index);
                    processing_nodes.push(potential_node.rch_index);
                    if (potential_node.lch_index <= 0 || potential_node.rch_index <= 0) {
                        LOG(FATAL);
                    }
                }
            }
        } else {
            processing_nodes.push(node.lch_index);
            processing_nodes.push(node.rch_index);
            if (node.lch_index <= 0 || node.rch_index <= 0) {
                LOG(FATAL);
            }
        }
    }
}


