//
// Created by liqinbin on 11/3/20.
//

#ifndef FEDTREE_HIST_CUT_H
#define FEDTREE_HIST_CUT_H

#include "FedTree/common.h"
#include "FedTree/dataset.h"
#include "tree.h"

class HistCut {
public:

    // The vales of cut points
    SyncArray<float_type> cut_points_val;
    // The number of accumulated cut points for current feature
    SyncArray<int> cut_col_ptr;
    // The feature id for current cut point
    SyncArray<int> cut_fid;

    HistCut() = default;

    HistCut(const HistCut &cut) {
        cut_points_val.copy_from(cut.cut_points_val);
        cut_col_ptr.copy_from(cut.cut_col_ptr);
    }

    // equally divide the feature range to get cut points
    // void get_cut_points(float_type feature_min, float_type feature_max, int max_num_bins, int n_instances);
    void get_cut_points_by_data_range(DataSet &dataset, int max_num_bins, int n_instances);
    void get_cut_points_fast(DataSet &dataset, int max_num_bins, int n_instances);
    void get_cut_points_by_n_instance(DataSet &dataset, int max_num_bins);
    void get_cut_points_by_feature_range(vector<vector<float>> f_range, int max_num_bins);
};

class RobustHistCut {
public:
    // The values of cut points
    vector<float_type> cut_points_val;
    // The number of accumulated cut points for current feature
    vector<int> cut_col_ptr;
    // The feature id for current cut point
    vector<int> cut_fid;
    // Number of instances in bins
    vector<vector<int>> n_instances_in_hist;

    RobustHistCut() = default;

    RobustHistCut(const RobustHistCut& cut): cut_points_val(cut.cut_points_val), cut_col_ptr(cut.cut_col_ptr),
    n_instances_in_hist(cut.n_instances_in_hist) { }

    void get_cut_points_by_feature_range_balanced(DataSet &dataset, int max_bin_size, int n_instances);
    void get_cut_points_by_instance(DataSet &dataset, int max_num_bins, int n_instances);

    [[nodiscard]] inline float_type get_cut_point_val(int fid, int bid) const {
        int feature_offset = cut_col_ptr[fid];
        return cut_points_val[feature_offset + bid];
    }

    [[nodiscard]] inline auto get_cut_point_val_itr(int fid, int bid) const {
        int feature_offset = cut_col_ptr[fid];
        return cut_points_val.begin() + feature_offset + bid;
    }

    [[nodiscard]] inline float_type get_cut_point_val(int bid) const {      // global bid
        return cut_points_val[bid];
    }

    [[nodiscard]] inline auto get_cut_point_val_itr(int bid) const {       // global bid
        return cut_points_val.begin() + bid;
    }
};


#endif //FEDTREE_HIST_CUT_H
