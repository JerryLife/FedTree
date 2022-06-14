//
// Created by liqinbin on 11/3/20.
//

#ifndef FEDTREE_HIST_CUT_H
#define FEDTREE_HIST_CUT_H

#include "FedTree/common.h"
#include "FedTree/dataset.h"
#include "tree.h"
#include "openssl/md5.h"
#include <random>

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
    // The values of cut points C1, C2, ..., Cn sorted in descending order.
    // The bins look like  (+inf, C1] (C1, C2] ... (Cn-1, Cn]
    vector<float_type> cut_points_val;
    // The number of accumulated cut points for current feature
    vector<int> cut_col_ptr;
    // The feature id for current cut point
    vector<int> cut_fid;
    // Number of instances in bins B1, B2, ..., Bn. Bi is the #intances in (Ci-1, Ci]
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

    static bool cut_value_hash_comp(const float_type v1, const float_type v2) {
        /**
         * Compare cut values by hash. The result depends entirely on v1 and v2, but in an almost random way.
         */
//        std::string fmt_v1 = string_format("%.4f", v1);
//        std::string fmt_v2 = string_format("%.4f", v2);
//        unsigned char md5_v1[MD5_DIGEST_LENGTH];
//        unsigned char md5_v2[MD5_DIGEST_LENGTH];
//        MD5((unsigned char *)fmt_v1.c_str(), fmt_v1.size(), md5_v1);
//        MD5((unsigned char *)fmt_v2.c_str(), fmt_v1.size(), md5_v2);
//        std::string md5_v1_str(reinterpret_cast<const char *>(md5_v1));
//        std::string md5_v2_str(reinterpret_cast<const char *>(md5_v2));
//        return md5_v1_str < md5_v2_str;
        auto v1_seed = (unsigned long) std::round(v1 * 10000);
        auto v2_seed = (unsigned long) std::round(v2 * 10000);
        std::mt19937 rng_v1{v1_seed};
        std::mt19937 rng_v2{v2_seed};
        std::uniform_int_distribution<unsigned> dist(std::mt19937::min(), std::mt19937::max());
        auto v1_value = dist(rng_v1);
        auto v2_value = dist(rng_v2);
        return v1_value < v2_value;
    }
};


#endif //FEDTREE_HIST_CUT_H
