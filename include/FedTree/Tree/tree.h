//
// Created by liqinbin on 10/13/20.
// The tree structure is referring to the design of ThunderGBM: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/tree.h
//

#ifndef FEDTREE_TREE_H
#define FEDTREE_TREE_H

#include "sstream"
#include "FedTree/syncarray.h"
#include "GBDTparam.h"


//class SplitPoint {
//public:
//    float_type gain;
//    GHPair fea_missing_gh;//missing gh in this segment
//    GHPair rch_sum_gh;//right child total gh (missing gh included if default2right)
//    bool default_right;
//    int nid;
//
//    //split condition
//    int split_fea_id;
//    float_type fval;//split on this feature value (for exact)
//    unsigned char split_bid;//split on this bin id (for hist)
//
//    SplitPoint() {
//        nid = -1;
//        split_fea_id = -1;
//        gain = 0;
//    }
//
//    friend std::ostream &operator<<(std::ostream &output, const SplitPoint &sp) {
//        output << sp.gain << "/" << sp.split_fea_id << "/" << sp.nid << "/" << sp.rch_sum_gh;
//        return output;
//    }
//};


class Tree{
public:
    struct TreeNode {
        int final_id;// node id after pruning, may not equal to node index
        int lch_index;// index of left child
        int rch_index;// index of right child
        int parent_index;// index of parent node
        float_type gain;// gain of splitting this node
        float_type base_weight;
        int split_feature_id;
        int pid;
        float_type split_value;
        unsigned char split_bid;
        bool default_right;
        bool is_leaf;
        bool is_valid;// non-valid nodes are those that are "children" of leaf nodes
        bool is_pruned;// pruned after pruning

        GHPair sum_gh_pair;
        int n_instances = 0; // number of instances inside the node.

        friend std::ostream &operator<<(std::ostream &os,
                                        const TreeNode &node);

        HOST_DEVICE void calc_weight(float_type lambda) {
            this->base_weight = -sum_gh_pair.g / (sum_gh_pair.h + lambda);
        }

        HOST_DEVICE bool splittable() const {
            return !is_leaf && is_valid;
        }

        TreeNode() = default;

        HOST_DEVICE TreeNode(const TreeNode& copy){
            final_id = copy.final_id;
            lch_index = copy.lch_index;
            rch_index = copy.rch_index;
            parent_index = copy.parent_index;
            gain = copy.gain;
            base_weight = copy.base_weight;
            split_feature_id = copy.split_feature_id;
            pid = copy.pid;
            split_value = copy.split_value;
            split_bid = copy.split_bid;
            default_right = copy.default_right;
            is_leaf = copy.is_leaf;
            is_valid = copy.is_valid;
            is_pruned = copy.is_pruned;
            sum_gh_pair.g = copy.sum_gh_pair.g;
            sum_gh_pair.h = copy.sum_gh_pair.h;
            n_instances = copy.n_instances;
        }

    };

    Tree() = default;

    Tree(const Tree &tree) {
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
        n_nodes_level = tree.n_nodes_level;
        final_depth = tree.final_depth;
    }

    Tree &operator=(const Tree &tree) {
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
        n_nodes_level = tree.n_nodes_level;
        final_depth = tree.final_depth;
        return *this;
    }

    void init_CPU(const GHPair sum_gh, const GBDTParam &param);

    void init_CPU(const SyncArray<GHPair> &gradients, const GBDTParam &param);

    virtual void init_structure(int depth);

    // TODO: GPU initialization 
    // void init2(const SyncArray<GHPair> &gradients, const GBDTParam &param);

    string dump(int depth) const;


    SyncArray<Tree::TreeNode> nodes;
    //n_nodes_level[i+1] - n_nodes_level[i] stores the number of nodes in level i
    vector<int> n_nodes_level;
    int final_depth;

    virtual void prune_self(float_type gamma);

    void compute_leaf_value();

protected:
    void preorder_traversal(int nid, int max_depth, int depth, string &s) const;

    virtual int try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count);

    virtual void reorder_nid();
};

struct DeltaTree : public Tree {
    struct DeltaNode : TreeNode {

        vector<int> potential_nodes_indices;    // the indices is sorted by the value of priority

        DeltaNode() = default;

        DeltaNode(const DeltaNode& copy) = default;

        DeltaNode &operator=(const DeltaNode& copy) {
            final_id = copy.final_id;
            lch_index = copy.lch_index;
            rch_index = copy.rch_index;
            parent_index = copy.parent_index;
            gain = copy.gain;
            base_weight = copy.base_weight;
            split_feature_id = copy.split_feature_id;
            pid = copy.pid;
            split_value = copy.split_value;
            split_bid = copy.split_bid;
            default_right = copy.default_right;
            is_leaf = copy.is_leaf;
            is_valid = copy.is_valid;
            is_pruned = copy.is_pruned;
            sum_gh_pair.g = copy.sum_gh_pair.g;
            sum_gh_pair.h = copy.sum_gh_pair.h;
            n_instances = copy.n_instances;
            potential_nodes_indices = copy.potential_nodes_indices;

            return *this;
        }

        inline bool is_robust() const { return potential_nodes_indices.empty(); }
    };

    DeltaTree() = default;

    DeltaTree(const DeltaTree& other) : Tree(other) {
        nodes = other.nodes;
        n_nodes_level = other.n_nodes_level;
        final_depth = other.final_depth;
    }

    DeltaTree &operator=(const DeltaTree &tree) {
        nodes = tree.nodes;
        n_nodes_level = tree.n_nodes_level;
        final_depth = tree.final_depth;
        return *this;
    }

    void init_CPU(const SyncArray<GHPair> &gradients, const DeltaBoostParam &param);

    void init_structure(int depth) override;

    void prune_self(float_type gamma) override;

    int try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count) override;

    void reorder_nid() override;

    vector<DeltaNode> nodes;    // contains all the nodes including potential nodes
};


#endif //FEDTREE_TREE_H
