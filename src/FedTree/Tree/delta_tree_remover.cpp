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
            node.sum_gh_pair.g += delta_gh_pairs[i].g;
            node.sum_gh_pair.h += delta_gh_pairs[i].h;
            node.gain.self_g += delta_gh_pairs[i].g;
            node.gain.self_h += delta_gh_pairs[i].h;

            // update missing_gh
            bool is_missing;
            float_type split_fval = get_val(indices[i], node.split_feature_id, &is_missing);
            if (is_missing) {
                node.gain.missing_g += delta_gh_pairs[i].g;
                node.gain.missing_h += delta_gh_pairs[i].h;
            }
        }
    }

    // update left or right Gh_Pair of parent & recalculate default direction
#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            auto &node = tree_ptr->nodes[node_id];

            if (node.parent_index == -1) continue;      // root node
            auto &parent_node = tree_ptr->nodes[node.parent_index];
            bool is_left_child = (parent_node.lch_index == node_id);
            if (is_left_child) {
                parent_node.gain.lch_g += delta_gh_pairs[i].g;
                parent_node.gain.lch_h += delta_gh_pairs[i].h;
            } else {
                parent_node.gain.rch_g += delta_gh_pairs[i].g;
                parent_node.gain.rch_h += delta_gh_pairs[i].h;
            }


        }
    }

    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            auto &node = tree_ptr->nodes[node_id];

            node.calc_weight(param.lambda);     // this lambda should be consistent with the training

            if (!node.is_leaf) {
                node.gain.cal_gain_value();     // calculate original gain value
                // recalculate default direction
                if (node.default_right) {
                    assert(node.gain.gain_value < 0);
                    DeltaTree::DeltaGain default_left_gain(node.gain);
                    default_left_gain.lch_g += node.gain.missing_g;
                    default_left_gain.lch_h += node.gain.missing_h;
                    default_left_gain.rch_g -= node.gain.missing_g;
                    default_left_gain.rch_h -= node.gain.missing_h;
                    default_left_gain.cal_gain_value();
                    if (fabs(default_left_gain.gain_value) > fabs(node.gain.gain_value)) {
                        // switch default direction
                        node.gain = default_left_gain;
                    }
                } else {
                    assert(node.gain.gain_value > 0);
                    DeltaTree::DeltaGain default_right_gain(node.gain);
                    default_right_gain.rch_g += node.gain.missing_g;
                    default_right_gain.rch_h += node.gain.missing_h;
                    default_right_gain.lch_g -= node.gain.missing_g;
                    default_right_gain.lch_h -= node.gain.missing_h;
                    default_right_gain.cal_gain_value();
                    if (fabs(default_right_gain.gain_value) > fabs(node.gain.gain_value)) {
                        // switch default direction
                        default_right_gain.gain_value = -default_right_gain.gain_value;
                        node.gain = default_right_gain;
                    }
                }
            }
        }
    }

    sort_potential_nodes_by_gain(0);


//
//
//    SyncArray<int> csr_col_idx(dataSet->csr_col_idx.size());
//    SyncArray<float_type> csr_val(dataSet->csr_val.size());
//    SyncArray<int> csr_row_ptr(dataSet->csr_row_ptr.size());
//    csr_col_idx.copy_from(dataSet->csr_col_idx.data(), dataSet->csr_col_idx.size());
//    csr_val.copy_from(dataSet->csr_val.data(), dataSet->csr_val.size());
//    csr_row_ptr.copy_from(dataSet->csr_row_ptr.data(), dataSet->csr_row_ptr.size());
//
//    const auto csr_col_idx_data = csr_col_idx.host_data();
//    const auto csr_val_data = csr_val.host_data();
//    const auto csr_row_ptr_data = csr_row_ptr.host_data();
//
//    auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
//                       bool *is_missing) -> float_type {
//        //binary search to get feature value
//        const int *left = row_idx;
//        const int *right = row_idx + row_len;
//
//        while (left != right) {
//            const int *mid = left + (right - left) / 2;
//            if (*mid == idx) {
//                *is_missing = false;
//                return row_val[mid - row_idx];
//            }
//            if (*mid > idx)
//                right = mid;
//            else left = mid + 1;
//        }
//        *is_missing = true;
//        return 0;
//    };
//
////    vector<int> modified_leaf_indices;
////#pragma omp parallel for
//    for (int i = 0; i < indices.size(); ++i) {
//        int id = indices[i];
//        GHPair delta_gh_pair = delta_gh_pairs[i];
//
//        const float_type gradient = delta_gh_pair.g;
//        const float_type hessian = delta_gh_pair.h;
//
//        int *col_idx = csr_col_idx_data + csr_row_ptr_data[id];
//        float_type *row_val = csr_val_data + csr_row_ptr_data[id];
//        int row_len = csr_row_ptr_data[id + 1] - csr_row_ptr_data[id];
//
//        std::queue<int> processing_nodes;
//        processing_nodes.push(0);    // start from root node
//        while (!processing_nodes.empty()) {
//            int nid = processing_nodes.front();
//            processing_nodes.pop();
//            auto& node = tree_ptr->nodes[nid];
//
//            if (!node.is_valid || node.is_robust()) {
//                continue;
//            }
//
//            if (node.is_leaf) {
//                // update leaf value
//                node.sum_gh_pair.g += gradient;
//                node.sum_gh_pair.h += hessian;
//                node.calc_weight(param.lambda);    // update node.base_weight
////                modified_leaf_indices.emplace_back(nid);
//            } else {
//                for (int j: node.potential_nodes_indices) {
//                    auto &potential_node = tree_ptr->nodes[j];
//
//                    // update the gain in each potential node
//                    potential_node.sum_gh_pair.g += gradient;
//                    potential_node.sum_gh_pair.h += hessian;
//                    potential_node.calc_weight(param.lambda);
//
//                    bool is_missing;
//                    float_type split_fval = get_val(col_idx, row_val, row_len, node.split_feature_id, &is_missing);
//                    if (split_fval < potential_node.split_value) {
//                        // goes left
//                        processing_nodes.push(potential_node.lch_index);
//                        potential_node.gain.delta_left_(gradient, hessian);
//                    }
//                    else {
//                        // goes right
//                        processing_nodes.push(potential_node.rch_index);
//                        potential_node.gain.delta_right_(gradient, hessian);
//                    }
//                }
//            }
//        }
//    }

//    sort_potential_nodes_by_gain(0);

//    return modified_leaf_indices;
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
