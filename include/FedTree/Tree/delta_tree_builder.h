//
// Created by HUSTW on 7/31/2021.
//

#include "hist_tree_builder.h"

#ifndef FEDTREE_DELTA_TREE_BUILDER_H
#define FEDTREE_DELTA_TREE_BUILDER_H

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

    void update_ins2node_id() override;

    void update_tree() override;

    void predict_in_training(int k) override;

    DeltaTree tree;
    DeltaBoostParam param;
};

#endif //FEDTREE_DELTA_TREE_BUILDER_H
