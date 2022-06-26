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

void DeltaTreeRemover::adjust_split_nbrs_by_indices(const vector<int>& adjusted_indices, const vector<GHPair>& root_delta_gh_pairs,
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

//    vector<vector<int>> nid_to_index_id(tree_ptr->nodes.size(), vector<int>());
//    for (int i = 0; i < indices.size(); ++i) {
//        for (auto node_id: ins2node_indices[i]) {
//            nid_to_index_id[node_id].push_back(i);
//        }
//    }

/**
 * Update gradients that are deleted
 */

// update the gain of all nodes according to ins2node_indices
    vector<vector<int>> updating_node_indices(adjusted_indices.size(), vector<int>(0));
#pragma omp parallel for
    for (int i = 0; i < adjusted_indices.size(); ++i) {
        updating_node_indices[i] = ins2node_indices[adjusted_indices[i]];
    }

    // update GH_pair of node and parent (parallel)
#pragma omp parallel for
    for (int i = 0; i < updating_node_indices.size(); ++i) {
        for (int node_id: updating_node_indices[i]) {
            // update self gh
            auto &node = tree_ptr->nodes[node_id];
#pragma omp atomic
            node.sum_gh_pair.g += root_delta_gh_pairs[i].g;
#pragma omp atomic
            node.sum_gh_pair.h += root_delta_gh_pairs[i].h;
#pragma omp atomic
            node.gain.self_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
            node.gain.self_h += root_delta_gh_pairs[i].h;

            // obtain feature value of instance adjusted_indices[i] in feature node.split_feature_id
            bool is_missing;
            float_type feature_val = get_val(adjusted_indices[i], node.split_feature_id, &is_missing);

            // update missing_gh
            if (is_missing) {
#pragma omp atomic
                node.gain.missing_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.gain.missing_h += root_delta_gh_pairs[i].h;
            }

            // update left gh and right gh (can be optimized because gain will later be updated)
            if (!node.is_leaf && feature_val < node.split_value) {
#pragma omp atomic
                node.gain.lch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.gain.lch_h += root_delta_gh_pairs[i].h;
            } else {
#pragma omp atomic
                node.gain.rch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.gain.rch_h += root_delta_gh_pairs[i].h;
            }

            // update split neighborhood
            for (int j = 0; j < node.split_nbr.split_bids.size(); ++j) {
#pragma omp atomic
                node.split_nbr.gain[j].self_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                node.split_nbr.gain[j].self_h += root_delta_gh_pairs[i].h;

                if (is_missing) {
#pragma omp atomic
                    node.split_nbr.gain[j].missing_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                    node.split_nbr.gain[j].missing_h += root_delta_gh_pairs[i].h;
                }

                if (!node.is_leaf && feature_val < node.split_nbr.split_vals[j]) {
#pragma omp atomic
                    node.split_nbr.gain[j].lch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                    node.split_nbr.gain[j].lch_h += root_delta_gh_pairs[i].h;
                } else {
#pragma omp atomic
                    node.split_nbr.gain[j].rch_g += root_delta_gh_pairs[i].g;
#pragma omp atomic
                    node.split_nbr.gain[j].rch_h += root_delta_gh_pairs[i].h;
                }
            }
        }
    }

//    /**
//    * Recalculate gain, adjust split_nbr, and record the old values.
//    */
//    vector<std::pair<int, int>> split_nbr_shift(tree_ptr->nodes.size());
//    // recalculate direction
//#pragma omp parallel for
//    for (int i = 0; i < updating_node_indices.size(); ++i) {
//        for (int node_id: updating_node_indices[i]) {
//            auto &node = tree_ptr->nodes[node_id];
//
//            node.calc_weight_(param.lambda);     // this lambda should be consistent with the training
//
//            if (!node.is_leaf) {
//                // recalculate the gain of each split neighbor
//                for (int j = 0; j < node.split_nbr.split_bids.size(); ++j) {
//                    node.split_nbr.gain[j].gain_value = node.split_nbr.gain[j].cal_gain_value();
//                }
//                // update the best gain
//                int old_best_idx = node.split_nbr.best_idx;
//                node.split_nbr.update_best_idx_();
//                int new_best_idx = node.split_nbr.best_idx;
//                node.gain = node.split_nbr.best_gain();
//                node.split_value = node.split_nbr.best_split_value();
//                node.split_bid = node.split_nbr.best_bid();
//
//                // store the change of best_idx
//                split_nbr_shift[node_id] = std::make_pair(old_best_idx, new_best_idx);
//
//            }
//        }
//    }

    /**
    * Update marginal gradients that are shifted
    */

    // inference from the root, update split_nbrs layer by layer
    size_t n_nodes_in_layer;
    vector<int> visit_node_indices = {0};

    // indices_in_nodes: indices in each node of a level
    vector<vector<int>> indices_in_nodes = {{}};
    vector<int> all_indices(dataSet->n_instances());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::copy_if(all_indices.begin(), all_indices.end(), std::back_inserter(indices_in_nodes[0]), [&](int i) {
        return !is_iid_removed[i];
    });
    vector<vector<int>> marginal_indices = {{}};
    vector<vector<GHPair>> marginal_gh = {{}};
    while (!visit_node_indices.empty()) {
        n_nodes_in_layer = visit_node_indices.size();
        vector<int> next_visiting_node_indices;
        vector<vector<int>> next_indices_in_nodes;
        vector<vector<int>> next_marginal_indices_in_nodes;
        vector<vector<GHPair>> next_marginal_gh_in_nodes;
        assert(n_nodes_in_layer == indices_in_nodes.size());

        // can try parallel
        for (int i = 0; i < n_nodes_in_layer; ++i) {
            int node_id = visit_node_indices[i];
            auto &node = tree_ptr->nodes[node_id];

            if (node_id == 51) {
                LOG(DEBUG);
            }

            const auto &indices_in_node = indices_in_nodes[i];
            const auto &marginal_indices_in_node = marginal_indices[i];
            const auto &marginal_gh_in_node = marginal_gh[i];

            // adjust the sum_g and sum_h according to marginal indices
#pragma omp parallel for
            for (int j = 0; j < marginal_indices_in_node.size(); ++j) {
                int iid = marginal_indices_in_node[j];
                bool is_missing;
                float_type feature_val = get_val(iid, node.split_feature_id, &is_missing);

                // update self gh
#pragma omp atomic
                node.sum_gh_pair.g += marginal_gh_in_node[j].g;
#pragma omp atomic
                node.sum_gh_pair.h += marginal_gh_in_node[j].h;
#pragma omp atomic
                node.gain.self_g += marginal_gh_in_node[j].g;
#pragma omp atomic
                node.gain.self_h += marginal_gh_in_node[j].h;

                // update left or right gh of split_nbr based on feature_val
                for (int k = 0; k < node.split_nbr.split_bids.size(); ++k) {
#pragma omp atomic
                    node.split_nbr.gain[k].self_g += marginal_gh_in_node[j].g;
#pragma omp atomic
                    node.split_nbr.gain[k].self_h += marginal_gh_in_node[j].h;

                    if (is_missing) {
#pragma omp atomic
                        node.split_nbr.gain[k].missing_g += marginal_gh_in_node[j].g;
#pragma omp atomic
                        node.split_nbr.gain[k].missing_h += marginal_gh_in_node[j].h;
                    }

                    if (!node.is_leaf && feature_val < node.split_nbr.split_vals[k]) {
#pragma omp atomic
                        node.split_nbr.gain[k].lch_g += marginal_gh_in_node[j].g;
#pragma omp atomic
                        node.split_nbr.gain[k].lch_h += marginal_gh_in_node[j].h;
                    } else {
#pragma omp atomic
                        node.split_nbr.gain[k].rch_g += marginal_gh_in_node[j].g;
#pragma omp atomic
                        node.split_nbr.gain[k].rch_h += marginal_gh_in_node[j].h;
                    }
                }
            }

            // re-select the best gain
            node.calc_weight_(param.lambda);     // this lambda should be consistent with the training
            if (node.is_leaf) continue;

            // recalculate the gain of each split neighbor
            for (int j = 0; j < node.split_nbr.split_bids.size(); ++j) {
                node.split_nbr.gain[j].gain_value = node.split_nbr.gain[j].cal_gain_value();
            }
            // update the best gain
            int old_best_idx = node.split_nbr.best_idx;
            node.split_nbr.update_best_idx_();
            node.gain = node.split_nbr.best_gain();
            float_type old_split_value = node.split_value;
            node.split_value = node.split_nbr.best_split_value();
            node.split_bid = node.split_nbr.best_bid();

            // recalculate default direction
            if (node.default_right) {
                node.gain.gain_value = -node.gain.gain_value;
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
                node.gain.gain_value = node.gain.gain_value;
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

            // recalculate indices to be adjusted in the next layer
            int cur_best_idx = node.split_nbr.best_idx;
            vector<GHPair> next_marginal_gh_left(dataSet->n_instances(), GHPair(0, 0, true));    // true means invalid to be filtered
            vector<GHPair> next_marginal_gh_right(dataSet->n_instances(), GHPair(0, 0, true));
            vector<int> next_marginal_indices_left(dataSet->n_instances(), -1);
            vector<int> next_marginal_indices_right(dataSet->n_instances(), -1);
            vector<int> next_indices_left(dataSet->n_instances(), -1);
            vector<int> next_indices_right(dataSet->n_instances(), -1);
            if (old_best_idx == cur_best_idx) {
                // simply determine all the instances go to left or right child
#pragma omp parallel for
                for (int j = 0; j < indices_in_node.size(); ++j) {
                    int iid = indices_in_node[j];
                    bool is_missing;
                    float_type feature_val = get_val(iid, node.split_feature_id, &is_missing);
                    if (feature_val < old_split_value) {
                        next_indices_left[iid] = iid;
                    } else {
                        next_indices_right[iid] = iid;
                    }
                }
            } else {
#pragma omp parallel for
                for (int j = 0; j < indices_in_node.size(); ++j) {
                    if (param.hash_sampling_round > 1 && !is_subset_indices[j]) continue;    // not trained in this tree
                    int iid = indices_in_node[j];
                    assert(!is_iid_removed[iid]);

                    auto lower_value = std::min(node.split_nbr.split_vals[old_best_idx], node.split_nbr.split_vals[cur_best_idx]);
                    auto upper_value =  std::max(node.split_nbr.split_vals[old_best_idx], node.split_nbr.split_vals[cur_best_idx]);
                    bool is_missing;
                    float_type feature_val = get_val(iid, node.split_feature_id, &is_missing);

                    if (feature_val < node.split_value) {
                        next_indices_left[iid] = iid;
                    } else {
                        next_indices_right[iid] = iid;
                    }
                    if (feature_val >= lower_value && feature_val < upper_value) {
                        // marginal instances
                        // |---right node-----|----marginal instances---|-----left node------|
                        // (the split indices is sorted in descending order of feature values)
                        next_marginal_indices_left[iid] = iid;
                        next_marginal_indices_right[iid] = iid;
                        if (old_best_idx < cur_best_idx) {
                            // move instances from left to right
                            next_marginal_gh_left[iid] = -gh_pairs[iid];
                            next_marginal_gh_right[iid] = gh_pairs[iid];
                        } else {
                            // move instances from right to left
                            next_marginal_gh_left[iid] = gh_pairs[iid];
                            next_marginal_gh_right[iid] = -gh_pairs[iid];
                        }
                    }
                }
            }
            GHPair left_acc1 = std::accumulate(next_marginal_gh_left.begin(), next_marginal_gh_left.end(), GHPair(), [](auto &a, auto &b){
                return GHPair(a.g + b.g, a.h + b.h);
            });
            GHPair right_acc1 = std::accumulate(next_marginal_gh_right.begin(), next_marginal_gh_right.end(), GHPair(), [](auto &a, auto &b){
                return GHPair(a.g + b.g, a.h + b.h);
            });

            // merge these the marginal gh in this node into the next_marginal_gh_left and next_marginal_gh_right
#pragma omp parallel for
            for (int j = 0; j < marginal_indices_in_node.size(); ++j) {
                int iid = marginal_indices_in_node[j];
                bool is_missing;
                float_type feature_val = get_val(iid, node.split_feature_id, &is_missing);

                if (feature_val < old_split_value) {
                    // this parent's marginal instance goes left
                    if (next_marginal_indices_left[iid] == -1) {
//#pragma omp atomic
                        next_marginal_indices_left[iid] = iid;
                        next_marginal_gh_left[iid] = marginal_gh_in_node[j];
                    } else {
#pragma omp atomic
                        next_marginal_gh_left[iid].g += marginal_gh_in_node[j].g;
#pragma omp atomic
                        next_marginal_gh_left[iid].h += marginal_gh_in_node[j].h;
                    }
                } else {
                    if (next_marginal_indices_right[iid] == -1) {
                        next_marginal_indices_right[iid] = iid;
                        next_marginal_gh_right[iid] = marginal_gh_in_node[j];
                    } else {
#pragma omp atomic
                        next_marginal_gh_right[iid].g += marginal_gh_in_node[j].g;
#pragma omp atomic
                        next_marginal_gh_right[iid].h += marginal_gh_in_node[j].h;
                    }
                }
            }

            GHPair left_acc2 = std::accumulate(next_marginal_gh_left.begin(), next_marginal_gh_left.end(), GHPair(), [](auto &a, auto &b){
                return GHPair(a.g + b.g, a.h + b.h);
            });
            GHPair right_acc2 = std::accumulate(next_marginal_gh_right.begin(), next_marginal_gh_right.end(), GHPair(), [](auto &a, auto &b){
                return GHPair(a.g + b.g, a.h + b.h);
            });

            // remove invalid values
            auto clean_gh_ = [](vector<GHPair>& ghs) {
                ghs.erase(std::remove_if(ghs.begin(), ghs.end(), [](GHPair &gh) {
                    return gh.encrypted;
                }), ghs.end());
            };
            auto clean_indices_ = [](vector<int>& indices) {
                indices.erase(std::remove_if(indices.begin(), indices.end(), [](int i) {
                    return i == -1;
                }), indices.end());
            };
            clean_gh_(next_marginal_gh_left);
            clean_gh_(next_marginal_gh_right);
            clean_indices_(next_indices_left);
            clean_indices_(next_indices_right);
            clean_indices_(next_marginal_indices_left);
            clean_indices_(next_marginal_indices_right);

            next_indices_in_nodes.emplace_back(next_indices_left);
            next_indices_in_nodes.emplace_back(next_indices_right);
            next_marginal_indices_in_nodes.emplace_back(next_marginal_indices_left);
            next_marginal_indices_in_nodes.emplace_back(next_marginal_indices_right);
            next_marginal_gh_in_nodes.emplace_back(next_marginal_gh_left);
            next_marginal_gh_in_nodes.emplace_back(next_marginal_gh_right);

            // add indices of left and right children
            assert(node.lch_index > 0 && node.rch_index > 0);
            next_visiting_node_indices.push_back(node.lch_index);
            next_visiting_node_indices.push_back(node.rch_index);
        }
        visit_node_indices = next_visiting_node_indices;
        indices_in_nodes = next_indices_in_nodes;
        marginal_indices = next_marginal_indices_in_nodes;
        marginal_gh = next_marginal_gh_in_nodes;
        next_visiting_node_indices.clear();
        next_indices_in_nodes.clear();
        next_marginal_indices_in_nodes.clear();
        next_marginal_gh_in_nodes.clear();
    }
}




