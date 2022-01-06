//
// Created by HUSTW on 7/31/2021.
//

#include "hist_tree_builder.h"
#include <utility>

#ifndef FEDTREE_DELTA_TREE_BUILDER_H
#define FEDTREE_DELTA_TREE_BUILDER_H

typedef std::pair<int, DeltaTree::DeltaGain> gain_pair;

class DeltaTreeBuilder: public HistTreeBuilder {
public:
    void init(DataSet &dataset, const DeltaBoostParam &param);

    void init_nocutpoints(DataSet &dataset, const DeltaBoostParam &param);

    vector<DeltaTree> build_delta_approximate(const SyncArray<GHPair> &gradients,
                                              std::vector<std::vector<int>>& ins2node_indices_in_tree,
                                              bool update_y_predict = true);

    void find_split(int level) override;

    void compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                      int *hist_fid, SyncArray<GHPair> &missing_gh,
                                      SyncArray<GHPair> &hist) override;

    void compute_gain_in_a_level(vector<DeltaTree::DeltaGain> &gain, int n_nodes_in_level, int n_bins, int *hist_fid,
            SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist, int n_column = 0);

    void get_split_points(vector<gain_pair> &best_idx_gain, int n_nodes_in_level, int *hist_fid,
                          SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist);

    void get_topk_gain_in_a_level(const vector<DeltaTree::DeltaGain> &gain, vector<vector<gain_pair>> &topk_idx_gain,
                                  int n_nodes_in_level, int n_bins, int k = 1);

    int get_threshold_gain_in_a_level(const vector<DeltaTree::DeltaGain> &gain, vector<vector<gain_pair>> &topk_idx_gain,
                                       int n_nodes_in_level, int n_bins, float_type min_diff, float_type max_range,
                                       const vector<int> &n_samples_in_nodes);

    void update_ins2node_id() override;

    void update_ins2node_indices();

    void update_tree() override;

    void predict_in_training(int k) override;

    void get_potential_split_points(const vector<vector<gain_pair>> &candidate_idx_gain,
                                    const int n_nodes_in_level,
                                    const int *hist_fid, SyncArray<GHPair> &missing_gh,
                                    SyncArray<GHPair> &hist, int level);

    int filter_potential_idx_gain(const vector<vector<gain_pair>>& candidate_idx_gain,
                                  vector<vector<gain_pair>>& potential_idx_gain,
                                  float_type quantized_width, int max_num_potential);

    void broadcast_potential_node_indices(int node_id);

    DeltaTree tree;
    DeltaBoostParam param;
    SyncArray<DeltaSplitPoint> sp;

    vector<int> num_nodes_per_level;    // number of nodes in each level, including potential nodes
    vector<vector<int>> ins2node_indices;   // each instance may be in multiple nodes

    vector<int> parent_indices;     // ID: the relative index of child in the layer
                                    // Value: the relative index of its parent in the layer
};

#endif //FEDTREE_DELTA_TREE_BUILDER_H
