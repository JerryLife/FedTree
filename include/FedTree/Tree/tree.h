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
//#include "VacuumFilter/vacuum.h"  // cannot include

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
        float_type split_value = 0.0;
        int split_bid;
        bool default_right = false;
        bool is_leaf = true;
        bool is_valid;// non-valid nodes are those that are "children" of leaf nodes
        bool is_pruned = false;// pruned after pruning

        GHPair sum_gh_pair;
        int n_instances = 0; // number of instances inside the node.

        friend std::ostream &operator<<(std::ostream &os,
                                        const TreeNode &node);

        HOST_DEVICE void calc_weight_(float_type lambda) {
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

        friend TreeNode tag_invoke(json::value_to_tag<TreeNode>, json::value const& v) {
            auto &o = v.as_object();

            TreeNode node;

            node.final_id = o.at("final_id").as_int64();
            node.lch_index = o.at("lch_index").as_int64();
            node.rch_index = o.at("rch_index").as_int64();
            node.parent_index = o.at("parent_index").as_int64();
            node.gain = o.at("gain").as_double();
            node.base_weight = o.at("base_weight").as_double();
            node.split_feature_id = o.at("split_feature_id").as_int64();
            node.pid = o.at("pid").as_int64();
            node.split_value = o.at("split_value").as_double();
            node.split_bid = o.at("split_bid").as_int64();
            node.default_right = o.at("default_right").as_bool();
            node.is_leaf = o.at("is_leaf").as_bool();
            node.is_valid = o.at("is_valid").as_bool();
            node.is_pruned = o.at("is_pruned").as_bool();
            node.sum_gh_pair.g = o.at("sum_gh_pair.g").as_double();
            node.sum_gh_pair.h = o.at("sum_gh_pair.h").as_double();
            node.n_instances = o.at("n_instances").as_int64();

            return node;
        }

        friend void tag_invoke(json::value_from_tag, json::value& v, TreeNode const& node) {
            v = json::object {
                    {"final_id", node.final_id},
                    {"lch_index", node.lch_index},
                    {"rch_index", node.rch_index},
                    {"parent_index", node.parent_index},
                    {"gain", node.gain},
                    {"base_weight", node.base_weight},
                    {"split_feature_id", node.split_feature_id},
                    {"pid", node.pid},
                    {"split_value", node.split_value},
                    {"split_bid", node.split_bid},
                    {"default_right", node.default_right},
                    {"is_leaf", node.is_leaf},
                    {"is_valid", node.is_valid},
                    {"is_pruned", node.is_pruned},
                    {"sum_gh_pair.g", node.sum_gh_pair.g},
                    {"sum_gh_pair.h", node.sum_gh_pair.h},
                    {"n_instances", node.n_instances},
            };
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

    friend Tree tag_invoke(json::value_to_tag<Tree>, json::value const& v) {
        auto &o = v.as_object();

        Tree tree;

        tree.nodes.load_from_vec(json::value_to<std::vector<TreeNode>>(v.at("nodes")));
        tree.n_nodes_level = json::value_to<std::vector<int>>(v.at("n_nodes_level"));
        tree.final_depth = v.at("final_depth").as_int64();

        return tree;
    }

    friend void tag_invoke(json::value_from_tag, json::value& v, Tree const& tree) {
        v = json::object {
                {"nodes", json::value_from(tree.nodes.to_vec())},
                {"n_nodes_level", json::value_from(tree.n_nodes_level)},
                {"final_depth", tree.final_depth}
        };
    }


protected:
    void preorder_traversal(int nid, int max_depth, int depth, string &s) const;

    virtual int try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count);

    virtual void reorder_nid();
};


//template<typename fp_t, int fp_len>
//class IndexFilter: public VacuumFilter<fp_t, fp_len> {
//
//};



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

        float_type ev_remain_gain = 0;
        float_type lch_g2 = 0;
        float_type rch_g2 = 0;
        float_type self_g2 = 0;
        float_type missing_g2 = 0;
        int n_instances = 0;
        int n_remove = 0;

        DeltaGain() = default;

        DeltaGain(const DeltaGain &deltaGain) = default;

        DeltaGain &operator=(const DeltaGain &other) = default;

        DeltaGain(float_type lchG, float_type lchH, float_type rchG, float_type rchH,
                  float_type selfG, float_type selfH, float_type missingG, float_type missingH, float_type lambda,
                  float_type lchG2, float_type rchG2, float_type self_g2, float_type missing_g2, int nInstances, int n_remove) :
                  lch_g(lchG), lch_h(lchH), rch_g(rchG), rch_h(rchH), self_g(selfG), self_h(selfH),
                missing_g(missingG), missing_h(missingH), lambda(lambda), lch_g2(lchG2), rch_g2(rchG2), self_g2(self_g2),
                missing_g2(missing_g2), n_instances(nInstances), n_remove(n_remove) {}

        DeltaGain(float_type gain_value, float_type lchG, float_type lchH, float_type rchG, float_type rchH,
                  float_type selfG, float_type selfH, float_type missingG, float_type missingH, float_type lambda,
                  float_type ev_remain_gain, float_type lchG2, float_type rchG2, float_type self_g2, float_type missing_g2,
                  int nInstances, int n_remove) :
                gain_value(gain_value), lch_g(lchG), lch_h(lchH), rch_g(rchG), rch_h(rchH), self_g(selfG), self_h(selfH),
                missing_g(missingG), missing_h(missingH), lambda(lambda), ev_remain_gain(ev_remain_gain),
                lch_g2(lchG2), rch_g2(rchG2), self_g2(self_g2),
                missing_g2(missing_g2), n_instances(nInstances), n_remove(n_remove) {}

//        DeltaGain(float_type gainValue, float_type lchG, float_type lchH, float_type rchG, float_type rchH,
//                  float_type selfG, float_type selfH, float_type missing_g, float_type missing_h, float_type lambda)
//                  : gain_value(gainValue), lch_g(lchG), lch_h(lchH), rch_g(rchG), rch_h(rchH),
//                  self_g(selfG), self_h(selfH), missing_g(missing_g), missing_h(missing_h), lambda(lambda) {}

        [[nodiscard]] float_type cal_gain_value(float_type min_child_weight = 1) const {
            if (lch_h >= min_child_weight && rch_h >= min_child_weight) {
                return std::max(0.f, (lch_g * lch_g) / (lch_h + lambda) + (rch_g * rch_g) / (rch_h + lambda) -
                                     (self_g * self_g) / (self_h + lambda));
            } else {
                return 0;
            }
        }

        [[nodiscard]] float_type cal_ev_remain_gain(float_type min_child_weight = 1) const {
            if (lch_h >= min_child_weight && rch_h >= min_child_weight) {
                double remove_ratio = 1. * n_remove / n_instances;
                double coef = (1 - 2 * remove_ratio + remove_ratio * remove_ratio - remove_ratio / n_instances);
                auto left_G = (remove_ratio * lch_g2 + coef * lch_g * lch_g) / ((1 - remove_ratio) * lch_h + lambda);
                auto right_G = (remove_ratio * rch_g2 + coef * rch_g * rch_g) / ((1 - remove_ratio) * rch_h + lambda);
                auto self_G =
                        (remove_ratio * self_g2 + coef * self_g * self_g) / ((1 - remove_ratio) * self_h + lambda);
                auto remain_gain = static_cast<float_type>(left_G + right_G - self_G);
                return std::max(0.f, remain_gain);
            } else {
                return 0;
            }
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
            ar & lch_g2;
            ar & rch_g2;
            ar & self_g2;
            ar & missing_g2;
            ar & n_instances;
            ar & n_remove;
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
                static_cast<float_type>(o.at("lambda").as_double()),
                static_cast<float_type>(o.at("ev_remain_gain").as_double()),
                static_cast<float_type>(o.at("lch_g2").as_double()),
                static_cast<float_type>(o.at("rch_g2").as_double()),
                static_cast<float_type>(o.at("self_g2").as_double()),
                static_cast<float_type>(o.at("missing_g2").as_double()),
                static_cast<int>(o.at("n_instances").as_int64()),
                static_cast<int>(o.at("n_remove").as_int64())
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
                    {"lambda", deltaGain.lambda},
                    {"ev_remain_gain", deltaGain.ev_remain_gain},
                    {"lch_g2", deltaGain.lch_g2},
                    {"rch_g2", deltaGain.rch_g2},
                    {"self_g2", deltaGain.self_g2},
                    {"missing_g2", deltaGain.missing_g2},
                    {"n_instances", deltaGain.n_instances},
                    {"n_remove", deltaGain.n_remove}
            };
        }
    };

    struct SplitNeighborhood {
        int fid = -1;
        int best_idx = -1;
        vector<int> split_bids;     // size nbr_size    // bin id in the feature, which is similar to node.split_bid
        vector<DeltaTree::DeltaGain> gain;  // size: nbr_size
        vector<float_type> split_vals;  // size nbr_size

        SplitNeighborhood() = default;

        SplitNeighborhood(const vector<int> &split_bids, int fid, const vector<DeltaTree::DeltaGain> &gain,
                          const vector<float_type> &split_vals)
                : split_bids(split_bids), fid(fid), gain(gain), split_vals(split_vals) {}

        SplitNeighborhood &operator=(const SplitNeighborhood &other) = default;

        void update_best_idx_() {
            if (gain.empty()) return;
            auto max_gain_itr = std::max_element(gain.begin(), gain.end(), [](const auto &a, const auto &b) {
                return std::abs(a.gain_value) < std::abs(b.gain_value);
            });
            best_idx = static_cast<int>(max_gain_itr - gain.begin());
        }

        [[nodiscard]] inline bool is_marginal(int bid) const {
            /**
             * @param bid: bin id in the feature
             */
             return split_bids[0] <= bid && bid < split_bids[split_bids.size() - 1];
        }

        [[nodiscard]] inline bool is_marginal(float_type value) const {
            /**
             * @param value: value in the split feature
             */
            return split_vals[0] > value && value >= split_vals[split_vals.size() - 1];
        }

        inline DeltaTree::DeltaGain best_gain() const {
            if (best_idx == -1 || gain.empty())
                return {};
            return gain[best_idx];
        }

        inline int best_bid() const {
            if (best_idx == -1 || split_bids.empty())
                return 0;
            return split_bids[best_idx];
        }

        inline float_type best_split_value() const {
            if (best_idx == -1 || split_vals.empty())
                return 0.0;
            return split_vals[best_idx];
        }

        template<class Archive> void serialize(Archive &ar, const unsigned int /*version*/) {
            ar & fid;
            ar & best_idx;
            ar & split_bids;
            ar & split_vals;
            ar & gain;
        }

        friend SplitNeighborhood tag_invoke(json::value_to_tag<SplitNeighborhood>, json::value const& v) {
            auto &o = v.as_object();

            SplitNeighborhood split_nbr;

            split_nbr.fid = static_cast<int>(v.at("fid").as_int64());
            split_nbr.best_idx = static_cast<int>(v.at("best_idx").as_int64());
            split_nbr.split_bids = json::value_to<std::vector<int>>(v.at("split_bids"));
            split_nbr.gain = json::value_to<std::vector<DeltaGain>>(v.at("gain"));
            split_nbr.split_vals = json::value_to<std::vector<float_type>>(v.at("split_vals"));
            return split_nbr;
        }

        friend void tag_invoke(json::value_from_tag, json::value& v, SplitNeighborhood const& split_nbr) {
            v = json::object {
                    {"fid", split_nbr.fid},
                    {"best_idx", split_nbr.best_idx},
                    {"split_bids", json::value_from(split_nbr.split_bids)},
                    {"gain", json::value_from(split_nbr.gain)},
                    {"split_vals", json::value_from(split_nbr.split_vals)},
            };
        }
    };

    struct DeltaNode : TreeNode {

        vector<int> potential_nodes_indices;    // the indices is sorted by the value of priority
        DeltaGain gain;     // hide the float_type gain
        float_type sum_g2 = 0.0;   // sum of g^2 and h^2 in this node
        SplitNeighborhood split_nbr;    // split neighborhood around split point

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
            sum_g2 = copy.sum_g2;
            n_instances = copy.n_instances;
            potential_nodes_indices = copy.potential_nodes_indices;
            split_nbr = copy.split_nbr;
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
            sum_g2 = copy.sum_g2;
            n_instances = copy.n_instances;
            potential_nodes_indices = copy.potential_nodes_indices;
            split_nbr = copy.split_nbr;
            return *this;
        }

        inline bool is_robust() const { return potential_nodes_indices.size() <= 1; }

//        inline bool is_prior() const { return potential_nodes_indices[0] == final_id; }

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
            ar & sum_g2;
            ar & n_instances;
            ar & potential_nodes_indices;
            ar & split_nbr;
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
            deltaNode.sum_g2 = o.at("sum_g2").as_double();
            deltaNode.split_nbr = json::value_to<SplitNeighborhood>(v.at("split_nbr"));
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
                    {"potential_nodes_indices", json::value_from(deltaNode.potential_nodes_indices)},
                    {"sum_g2", deltaNode.sum_g2},
                    {"split_nbr", json::value_from(deltaNode.split_nbr)}
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
