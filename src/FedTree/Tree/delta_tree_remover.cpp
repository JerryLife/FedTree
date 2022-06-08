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
    adjust_split_nbrs_by_indices(indices, gh_pair_vec, true);
}




[[deprecated]]
void DeltaTreeRemover::adjust_gradients_by_indices(const vector<int>& indices, const vector<GHPair>& delta_gh_pairs) {
    /**
     * @param id: the indices of sample to be adjusted gradients
     * @param delta_gh_pair: gradient and hessian to be subtracted from the tree
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

            node.calc_weight_(param.lambda);     // this lambda should be consistent with the training

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

[[deprecated]]
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

void DeltaTreeRemover::adjust_split_nbrs_by_indices(const vector<int>& indices, const vector<GHPair>& delta_gh_pairs,
                                                    bool remove_n_ins) {
    /**
     * @param id: the indices of sample to be adjusted gradients
     * @param delta_gh_pair: gradient and hessian to be subtracted from the tree
     * @param remove_n_ins: whether to remove n_instances from visited nodes
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

    vector<vector<int>> nid_to_index_id(tree_ptr->nodes.size(), vector<int>());
    for (int i = 0; i < indices.size(); ++i) {
        for (auto node_id: ins2node_indices[i]) {
            nid_to_index_id[node_id].push_back(i);
        }
    }


    // inference from the root, update split_nbrs layer by layer
    size_t n_nodes_in_layer;
    vector<int> visit_node_indices = {0};

    // iid_indices_in_nodes: iid indices in each node of a level
    vector<vector<int>> iid_indices_in_nodes(1, vector<int>(indices.size()));
    // initialize with all the instances to be adjusted (root node)
    std::iota(iid_indices_in_nodes[0].begin(), iid_indices_in_nodes[0].end(), 0);
    while (!visit_node_indices.empty()) {
        n_nodes_in_layer = visit_node_indices.size();
        vector<vector<int>> next_iid_indices_in_nodes;
        assert(n_nodes_in_layer == iid_indices_in_nodes.size());

        // can try parallel
        for (int i = 0; i < n_nodes_in_layer; ++i) {
            int node_id = visit_node_indices[i];
            auto &node = tree_ptr->nodes[node_id];
            auto &iid_indices = iid_indices_in_nodes[i];

            for (int j: iid_indices) {
                if (remove_n_ins) {
                    node.n_instances -= 1;
                    node.gain.n_instances -= 1;
                }

                if (node.parent_index == -1) {
                    // root node
                    node.sum_gh_pair.g += delta_gh_pairs[j].g;
                    node.sum_gh_pair.h += delta_gh_pairs[j].h;

                    if (node.is_leaf) continue;

                    // update missing_gh
                    bool is_missing;
                    float_type feature_val = get_val(indices[j], node.split_feature_id, &is_missing);

                    // update all the neighbors
                    for (int k = 0; k < node.split_nbr.split_bids.size(); ++k) {
                        if (is_missing) {
                            node.split_nbr.gain[k].missing_g += delta_gh_pairs[j].g;
                            node.split_nbr.gain[k].missing_h += delta_gh_pairs[j].h;
                        }

                        node.split_nbr.gain[k].self_g += delta_gh_pairs[j].g;
                        node.split_nbr.gain[k].self_h += delta_gh_pairs[j].h;
                        if (feature_val < node.split_nbr.split_vals[k]) {
                            node.split_nbr.gain[k].lch_g += delta_gh_pairs[j].g;
                            node.split_nbr.gain[k].lch_h += delta_gh_pairs[j].h;
                        } else {
                            node.split_nbr.gain[k].rch_g += delta_gh_pairs[j].g;
                            node.split_nbr.gain[k].rch_h += delta_gh_pairs[j].h;
                        }
                    }
                } else {
                    // non-root nodes
                    assert(node.parent_index >= 0);

                    // obtain self.gh_pair from parent
                    const auto &parent_node = tree_ptr->nodes[node.parent_index];
                    if (tree_ptr->is_left_child(node_id)) {
                        // this node is the left child of its parent
                        node.sum_gh_pair.g = parent_node.gain.lch_g;
                        node.sum_gh_pair.h = parent_node.gain.lch_h;
                        for (int k = 0; k < node.split_nbr.split_bids.size(); ++k) {
                            node.split_nbr.gain[k].self_g = parent_node.gain.lch_g;
                            node.split_nbr.gain[k].self_h = parent_node.gain.lch_h;
                        }
                    } else {
                        // this node is the right child of its parent
                        node.sum_gh_pair.g = parent_node.gain.rch_g;
                        node.sum_gh_pair.h = parent_node.gain.rch_h;
                        for (int k = 0; k < node.split_nbr.split_bids.size(); ++k) {
                            node.split_nbr.gain[k].self_g = parent_node.gain.rch_g;
                            node.split_nbr.gain[k].self_h = parent_node.gain.rch_h;
                        }
                    }

                    if (node.is_leaf) continue;

                    bool is_missing;
                    float_type feature_val = get_val(indices[j], node.split_feature_id, &is_missing);

                    // update all the neighbors
                    for (int k = 0; k < node.split_nbr.split_bids.size(); ++k) {
                        if (is_missing) {
                            node.split_nbr.gain[k].missing_g += delta_gh_pairs[j].g;
                            node.split_nbr.gain[k].missing_h += delta_gh_pairs[j].h;
                        }
                        if (feature_val < node.split_nbr.split_vals[k]) {
                            node.split_nbr.gain[k].lch_g += delta_gh_pairs[j].g;
                            node.split_nbr.gain[k].lch_h += delta_gh_pairs[j].h;
                        } else {
                            node.split_nbr.gain[k].rch_g += delta_gh_pairs[j].g;
                            node.split_nbr.gain[k].rch_h += delta_gh_pairs[j].h;
                        }
                    }
                }
            }

            if (node.is_leaf) {
                node.calc_weight_(param.lambda);
                continue;
            }

            // select new best gain
            node.split_nbr.update_best_idx_();
            node.gain = node.split_nbr.best_gain();
            node.split_bid = node.split_nbr.best_bid();
            node.split_value = node.split_nbr.best_split_value();

            // recalculate default direction
            if (node.default_right) {
                node.gain.gain_value = -node.gain.cal_gain_value();
                assert(node.gain.gain_value <= 0);
                DeltaTree::DeltaGain default_left_gain(node.gain);
                default_left_gain.lch_g += node.gain.missing_g;
                default_left_gain.lch_h += node.gain.missing_h;
                default_left_gain.rch_g -= node.gain.missing_g;
                default_left_gain.rch_h -= node.gain.missing_h;
                default_left_gain.gain_value = default_left_gain.cal_gain_value();
                if (ft_ge(fabs(default_left_gain.gain_value), fabs(node.gain.gain_value), 1e-2)) {
                    // switch default direction to left (marginal default left)
                    node.gain = default_left_gain;
                    node.default_right = false;
                }
            } else {
                node.gain.gain_value = node.gain.cal_gain_value();
                assert(node.gain.gain_value >= 0);
                DeltaTree::DeltaGain default_right_gain(node.gain);
                default_right_gain.rch_g += node.gain.missing_g;
                default_right_gain.rch_h += node.gain.missing_h;
                default_right_gain.lch_g -= node.gain.missing_g;
                default_right_gain.lch_h -= node.gain.missing_h;
                default_right_gain.gain_value = -default_right_gain.cal_gain_value();
                if (!ft_ge(fabs(node.gain.gain_value), fabs(default_right_gain.gain_value), 1e-2)) {
                    // switch default direction to right (marginal default left)
                    default_right_gain.gain_value = -default_right_gain.gain_value;
                    node.gain = default_right_gain;
                    node.default_right = true;
                }
            }

            // recalculate iid_indices of next layer (split_value may be changed)
            vector<int> next_iid_indices_left;
            vector<int> next_iid_indices_right;
            for (int j: iid_indices) {
                bool is_missing;
                float_type feature_val = get_val(indices[j], node.split_feature_id, &is_missing);
                bool to_left;
                if (is_missing) {
                    to_left = !node.default_right;
                } else {
                    to_left = feature_val < node.split_value;
                }

                if (to_left) {
                    next_iid_indices_left.push_back(j);
                } else {
                    next_iid_indices_right.push_back(j);
                }
            }
            next_iid_indices_in_nodes.emplace_back(next_iid_indices_left);
            next_iid_indices_in_nodes.emplace_back(next_iid_indices_right);

            // add indices of left and right children
            assert(node.lch_index > 0 && node.rch_index > 0);
            visit_node_indices.push_back(node.lch_index);
            visit_node_indices.push_back(node.rch_index);
        }
        visit_node_indices.erase(visit_node_indices.begin(), visit_node_indices.begin() + n_nodes_in_layer);
        iid_indices_in_nodes = next_iid_indices_in_nodes;
        next_iid_indices_in_nodes.clear();
    }
}




