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
            node.calc_weight_(param.lambda);    // update node.base_weight
        } else {
            for (int i = 0; i < node.potential_nodes_indices.size(); ++i) {
                // update the gain in each potential node
                node.sum_gh_pair.g -= gradient;
                node.sum_gh_pair.h -= hessian;
            }

            // sort the nodes by descending order of gain


            processing_nodes.push(node.lch_index);
            processing_nodes.push(node.rch_index);
        }

    }

    return false;
}
