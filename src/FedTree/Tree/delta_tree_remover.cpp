//
// Created by HUSTW on 8/17/2021.
//

#include <algorithm>
#include <queue>

#include "FedTree/Tree/delta_tree_remover.h"

bool DeltaTreeRemover::remove_sample_by_id(int id) {
    /**
     * @param id: the index of sample to be removed from the tree
     * @return : true when a removal is successful; false when failing to remove and a retrain is needed
     */

    const float_type gradient = gh_pairs[id].g;
    const float_type hessian = gh_pairs[id].h;

    SyncArray<int> csr_col_idx(dataSet->csr_col_idx.size());
    SyncArray<float_type> csr_val(dataSet->csr_val.size());
    SyncArray<int> csr_row_ptr(dataSet->csr_row_ptr.size());
    csr_col_idx.copy_from(dataSet->csr_col_idx.data(), dataSet->csr_col_idx.size());
    csr_val.copy_from(dataSet->csr_val.data(), dataSet->csr_val.size());
    csr_row_ptr.copy_from(dataSet->csr_row_ptr.data(), dataSet->csr_row_ptr.size());

    auto csr_col_idx_data = csr_col_idx.host_data();
    auto csr_val_data = csr_val.host_data();
    auto csr_row_ptr_data = csr_row_ptr.host_data();

    int *col_idx = csr_col_idx_data + csr_row_ptr_data[id];
    float_type *row_val = csr_val_data + csr_row_ptr_data[id];
    int row_len = csr_row_ptr_data[id + 1] - csr_row_ptr_data[id];

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

    std::queue<int> processing_nodes;
    processing_nodes.push(0);    // start from root node
    while (!processing_nodes.empty()) {
        int nid = processing_nodes.front();
        processing_nodes.pop();
        auto& node = tree.nodes[nid];

        if (!node.is_valid || node.is_robust()) {
            continue;
        }

        if (node.is_leaf) {
            // update leaf value
            node.sum_gh_pair.g -= gradient;
            node.sum_gh_pair.h -= hessian;
            node.calc_weight(param.lambda);    // update node.base_weight
        } else {
            for (int i: node.potential_nodes_indices) {
                auto &potential_node = tree.nodes[i];

                // update the gain in each potential node
                potential_node.sum_gh_pair.g -= gradient;
                potential_node.sum_gh_pair.h -= hessian;
                potential_node.calc_weight(param.lambda);

                bool is_missing;
                float_type split_fval = get_val(col_idx, row_val, row_len, node.split_feature_id, &is_missing);
                if (split_fval < potential_node.split_value) {
                    // goes left
                    processing_nodes.push(potential_node.lch_index);
                    potential_node.gain.delta_left_(gradient, hessian);
                }
                else {
                    // goes right
                    processing_nodes.push(node.rch_index);
                    potential_node.gain.delta_right(gradient, hessian);
                }
            }

            // sort the nodes by descending order of gain
            std::sort(node.potential_nodes_indices.begin(), node.potential_nodes_indices.end(),
                      [&](int i, int j){
                return tree.nodes[i].gain.gain_value > tree.nodes[j].gain.gain_value;
            });

            // sync the order through potential nodes
            for (int i: node.potential_nodes_indices) {
                auto &potential_node = tree.nodes[i];
                potential_node.potential_nodes_indices = node.potential_nodes_indices;
            }
        }

    }

    return false;
}
