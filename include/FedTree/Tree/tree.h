//
// Created by liqinbin on 10/13/20.
// The tree structure is referring to the design of ThunderGBM: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/tree.h
//

#ifndef FEDTREE_TREE_H
#define FEDTREE_TREE_H

#include "boost/serialization/vector.hpp"
#include "boost/serialization/base_object.hpp"
#include <boost/json.hpp>

#include "sstream"
#include "FedTree/syncarray.h"
#include "GBDTparam.h"

namespace json = boost::json;


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
        int lch_index = -1;// index of left child
        int rch_index = -1;// index of right child
        int parent_index;// index of parent node
        float_type gain = 0;// gain of splitting this node
        float_type base_weight;
        int split_feature_id;
        int pid;
        float_type split_value;
        unsigned char split_bid;
        bool default_right = false;
        bool is_leaf = true;
        bool is_valid;// non-valid nodes are those that are "children" of leaf nodes
        bool is_pruned = false;// pruned after pruning

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
    // can be edited
    struct DeltaGain {
        float_type gain_value = 0;
        float_type lch_g = 0;
        float_type lch_h = 0;
        float_type rch_g = 0;
        float_type rch_h = 0;
        float_type self_g = 0;
        float_type self_h = 0;
        float_type missing_g = 0;
        float_type missing_h = 0;
        float_type lambda = 0;

        DeltaGain() = default;

        DeltaGain(float_type gainValue, float_type lchG, float_type lchH, float_type rchG, float_type rchH,
                  float_type selfG, float_type selfH, float_type missing_g, float_type missing_h, float_type lambda)
                  : gain_value(gainValue), lch_g(lchG), lch_h(lchH), rch_g(rchG), rch_h(rchH),
                  self_g(selfG), self_h(selfH), missing_g(missing_g), missing_h(missing_h), lambda(lambda) {}

        float_type cal_gain_value() const {
            return std::max(0.f, (lch_g * lch_g) / (lch_h + lambda) + (rch_g * rch_g) / (rch_h + lambda) -
                   (self_g * self_g) / (self_h + lambda));
        }

        void delta_left_(float_type gradient, float_type hessian) {
            lch_g += gradient;
            lch_h += hessian;
            self_g += gradient;
            self_h += hessian;
            gain_value = cal_gain_value();
        }

        void delta_right_(float_type gradient, float_type hessian) {
            rch_g += gradient;
            rch_h += hessian;
            self_g += gradient;
            self_h += hessian;
            gain_value = cal_gain_value();
        }

        friend class boost::serialization::access;

        template<class Archive> void serialize(Archive &ar, const unsigned int /*version*/) {
            ar & gain_value;
            ar & lch_g;
            ar & lch_h;
            ar & rch_g;
            ar & rch_h;
            ar & self_g;
            ar & self_h;
            ar & missing_g;
            ar & missing_h;
            ar & lambda;
        }

        friend DeltaGain tag_invoke(json::value_to_tag<DeltaGain>, json::value const& v) {
            auto &o = v.as_object();
            return {
                static_cast<float_type>(o.at("gain_value").as_double()),
                static_cast<float_type>(o.at("lch_g").as_double()),
                static_cast<float_type>(o.at("lch_h").as_double()),
                static_cast<float_type>(o.at("rch_g").as_double()),
                static_cast<float_type>(o.at("rch_h").as_double()),
                static_cast<float_type>(o.at("self_g").as_double()),
                static_cast<float_type>(o.at("self_h").as_double()),
                static_cast<float_type>(o.at("missing_g").as_double()),
                static_cast<float_type>(o.at("missing_h").as_double()),
                static_cast<float_type>(o.at("lambda").as_double())
            };
        }

        friend void tag_invoke(json::value_from_tag, json::value& v, DeltaGain const& deltaGain) {
            v = json::object {
                    {"gain_value", deltaGain.gain_value},
                    {"lch_g", deltaGain.lch_g},
                    {"lch_h", deltaGain.lch_h},
                    {"rch_g", deltaGain.rch_g},
                    {"rch_h", deltaGain.rch_h},
                    {"self_g", deltaGain.self_g},
                    {"self_h", deltaGain.self_h},
                    {"missing_g", deltaGain.missing_g},
                    {"missing_h", deltaGain.missing_h},
                    {"lambda", deltaGain.lambda}
            };
        }
    };

    struct DeltaNode : TreeNode {

        vector<int> potential_nodes_indices;    // the indices is sorted by the value of priority
        DeltaGain gain;     // hide the float_type gain

        DeltaNode() = default;

        DeltaNode(const DeltaNode& copy) {
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
        }

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

        inline bool is_robust() const { return potential_nodes_indices.size() <= 1; }

//        size_t to_chars(char* bytes) {
//            char buf[sizeof(size_t) + potential_nodes_indices.size() * sizeof(int) + sizeof(DeltaNode)];
//            size_t potential_node_size = potential_nodes_indices.size();
//            memcpy(buf, &potential_node_size, sizeof(size_t));
//            memcpy(buf + sizeof(size_t), potential_nodes_indices.data(), potential_node_size * sizeof(int));
//            memcpy(buf + sizeof(size_t) + potential_nodes_indices.size() * sizeof(int), this, sizeof(DeltaNode));
//        }
//
//        DeltaNode from_chars(char *bytes, size_t len);
    private:
        friend class boost::serialization::access;
        template<class Archive> void serialize(Archive &ar, const unsigned int version) {
            ar & final_id;
            ar & lch_index;
            ar & rch_index;
            ar & parent_index;
            ar & gain;
            ar & base_weight;
            ar & split_feature_id;
            ar & pid;
            ar & split_value;
            ar & split_bid;
            ar & default_right;
            ar & is_leaf;
            ar & is_valid;
            ar & is_pruned;
            ar & sum_gh_pair.g;
            ar & sum_gh_pair.h;
            ar & n_instances;
            ar & potential_nodes_indices;
        }

        friend DeltaNode tag_invoke(json::value_to_tag<DeltaNode>, json::value const& v) {
            auto &o = v.as_object();

            DeltaNode deltaNode;

            deltaNode.final_id = o.at("final_id").as_int64();
            deltaNode.lch_index = o.at("lch_index").as_int64();
            deltaNode.rch_index = o.at("rch_index").as_int64();
            deltaNode.parent_index = o.at("parent_index").as_int64();
            deltaNode.gain = json::value_to<DeltaGain>(o.at("gain"));
            deltaNode.base_weight = o.at("base_weight").as_double();
            deltaNode.split_feature_id = o.at("split_feature_id").as_int64();
            deltaNode.pid = o.at("pid").as_int64();
            deltaNode.split_value = o.at("split_value").as_double();
            deltaNode.split_bid = o.at("split_bid").as_int64();
            deltaNode.default_right = o.at("default_right").as_bool();
            deltaNode.is_leaf = o.at("is_leaf").as_bool();
            deltaNode.is_valid = o.at("is_valid").as_bool();
            deltaNode.is_pruned = o.at("is_pruned").as_bool();
            deltaNode.sum_gh_pair.g = o.at("sum_gh_pair.g").as_double();
            deltaNode.sum_gh_pair.h = o.at("sum_gh_pair.h").as_double();
            deltaNode.n_instances = o.at("n_instances").as_int64();
            deltaNode.potential_nodes_indices = boost::json::value_to<std::vector<int>>(o.at("potential_nodes_indices"));

            return deltaNode;
        }

        friend void tag_invoke(json::value_from_tag, json::value& v, DeltaNode const& deltaNode) {
            v = json::object {
                    {"final_id", deltaNode.final_id},
                    {"lch_index", deltaNode.lch_index},
                    {"rch_index", deltaNode.rch_index},
                    {"parent_index", deltaNode.parent_index},
                    {"gain", json::value_from(deltaNode.gain)},
                    {"base_weight", deltaNode.base_weight},
                    {"split_feature_id", deltaNode.split_feature_id},
                    {"pid", deltaNode.pid},
                    {"split_value", deltaNode.split_value},
                    {"split_bid", deltaNode.split_bid},
                    {"default_right", deltaNode.default_right},
                    {"is_leaf", deltaNode.is_leaf},
                    {"is_valid", deltaNode.is_valid},
                    {"is_pruned", deltaNode.is_pruned},
                    {"sum_gh_pair.g", deltaNode.sum_gh_pair.g},
                    {"sum_gh_pair.h", deltaNode.sum_gh_pair.h},
                    {"n_instances", deltaNode.n_instances},
                    {"potential_nodes_indices", json::value_from(deltaNode.potential_nodes_indices)}
            };
        }
    };

    DeltaTree() = default;

    DeltaTree(const DeltaTree& other) {
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

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive &ar, const unsigned int version) {
        ar & nodes;
        ar & n_nodes_level;
        ar & final_depth;
    }

    friend DeltaTree tag_invoke(json::value_to_tag<DeltaTree>, json::value const& v) {
        auto &o = v.as_object();

        DeltaTree deltaTree;

        deltaTree.nodes = json::value_to<std::vector<DeltaNode>>(v.at("nodes"));
        deltaTree.n_nodes_level = json::value_to<std::vector<int>>(v.at("n_nodes_level"));
        deltaTree.final_depth = v.at("final_depth").as_int64();

        return deltaTree;
    }

    friend void tag_invoke(json::value_from_tag, json::value& v, DeltaTree const& deltaTree) {
        v = json::object {
                {"nodes", json::value_from(deltaTree.nodes)},
                {"n_nodes_level", json::value_from(deltaTree.n_nodes_level)},
                {"final_depth", deltaTree.final_depth}
        };
    }

};


#endif //FEDTREE_TREE_H
