//
// Created by HUSTW on 7/31/2021.
//

#include "hist_tree_builder.h"
#include <utility>

#ifndef FEDTREE_DELTA_TREE_BUILDER_H
#define FEDTREE_DELTA_TREE_BUILDER_H

typedef std::pair<int, float_type> gain_pair;

class DeltaTreeBuilder: public HistTreeBuilder {
public:
    void init(DataSet &dataset, const DeltaBoostParam &param);

    void init_nocutpoints(DataSet &dataset, const DeltaBoostParam &param);

    vector<DeltaTree> build_delta_approximate(const SyncArray<GHPair> &gradients, bool update_y_predict = true);

    void find_split(int level) override;

    void compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                      int *hist_fid, SyncArray<GHPair> &missing_gh,
                                      SyncArray<GHPair> &hist) override;

    void compute_gain_in_a_level(SyncArray<float_type> &gain, int n_nodes_in_level, int n_bins, int *hist_fid,
            SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist, int n_column = 0) override;

    void get_split_points(SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int *hist_fid,
                          SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist) override;

    void get_topk_gain_in_a_level(const SyncArray<float_type> &gain, vector<vector<gain_pair>> &topk_idx_gain,
                                  int n_nodes_in_level, int n_bins, int k = 1);

    void update_ins2node_id() override;

    void update_tree() override;

    void predict_in_training(int k) override;

    void get_potential_split_points(const vector<vector<gain_pair>> &candidate_idx_gain,
                                    const int n_nodes_in_level,
                                    const int *hist_fid, SyncArray<GHPair> &missing_gh,
                                    SyncArray<GHPair> &hist, int level);

    int filter_potential_idx_gain(const vector<vector<gain_pair>>& candidate_idx_gain,
                                  vector<vector<gain_pair>>& potential_idx_gain,
                                  float_type quantized_width, int max_num_potential);

    DeltaTree tree;
    DeltaBoostParam param;

    vector<int> num_nodes_per_level;    // number of nodes in each level, including potential nodes
    vector<vector<int>> ins2node_indices;   // each instance may be in multiple nodes
};

#endif //FEDTREE_DELTA_TREE_BUILDER_H
