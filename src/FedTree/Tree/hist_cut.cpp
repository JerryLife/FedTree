// created by Tianyuan on 12/1/20

#include "FedTree/Tree/hist_cut.h"
#include "FedTree/util/device_lambda.h"
#include "FedTree/util/cub_wrapper.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include "thrust/unique.h"
#include "thrust/execution_policy.h"
#include "FedTree/common.h"
#include <numeric>


void HistCut::get_cut_points_by_data_range(DataSet &dataset, int max_num_bins, int n_instances){
    int n_column = dataset.n_features();
    SyncArray<float> unique_vals(n_column * n_instances);
    SyncArray<int> temp_row_ptr(n_column + 1);

    SyncArray<int> temp_params(2); //[num_cut_points, max_num_bins]
    int h_temp_params[2] = {0, max_num_bins};
    temp_params.copy_from(h_temp_params, 2);

    auto csc_val_data = &dataset.csc_val[0]; //CPU version; need conversion to syncarray pointer for GPU processing
    auto csc_col_ptr_data = &dataset.csc_col_ptr[0]; //CPU version; need conversion to syncarray pointer for GPU processing
    auto unique_vals_data = unique_vals.host_data();
    auto temp_row_ptr_data = temp_row_ptr.host_data();
    auto temp_params_data = temp_params.host_data();

#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        // copy data value from csc array to unique array
        int col_start = csc_col_ptr_data[fid];
        int col_len = csc_col_ptr_data[fid + 1] - col_start;

        auto val_data = csc_val_data + col_start;
        auto unique_start = unique_vals_data + fid * n_instances;  // notice here

        int unique_len = thrust::unique_copy(thrust::host, val_data, val_data + col_len, unique_start) - unique_start;
        int n_cp = (unique_len <= temp_params_data[1]) ? unique_len : temp_params_data[1];
        temp_row_ptr_data[fid + 1] = unique_len;
        // atomicAdd(&tmp_params_data[0], n_cp);
#pragma omp atomic
        temp_params_data[0] += n_cp;
    }

    // merge the cut points
    temp_params_data = temp_params.host_data();
    cut_points_val.resize(temp_params_data[0]);
    cut_col_ptr.resize(n_column + 1);
    cut_fid.resize(temp_params_data[0]);

    cut_col_ptr.copy_from(temp_row_ptr);
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    temp_row_ptr_data = temp_row_ptr.host_data();
    for(int i = 1; i < (n_column + 1); i++) {
        if(temp_row_ptr_data[i] <= temp_params_data[1])
            cut_col_ptr_data[i] += cut_col_ptr_data[i-1];
        else
            cut_col_ptr_data[i] = cut_col_ptr_data[i-1] + max_num_bins;
    }

    auto cut_point_val_data = cut_points_val.host_data();
    temp_row_ptr_data = temp_row_ptr.host_data();
    cut_col_ptr_data = cut_col_ptr.host_data();
    unique_vals_data = unique_vals.host_data();

#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        for (int i = cut_col_ptr_data[fid]; i < cut_col_ptr_data[fid + 1]; i++) {
            int unique_len = temp_row_ptr_data[fid + 1];
            int unique_idx = i - cut_col_ptr_data[fid];
            int cp_idx = (unique_len <= temp_params_data[1]) ? unique_idx : (unique_len / temp_params_data[1] *
                                                                             unique_idx);
            cut_point_val_data[i] = unique_vals_data[fid * n_instances + cp_idx];
        }
    }

    auto cut_fid_data = cut_fid.host_data();
#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        for (int i = cut_col_ptr_data[fid]; i < cut_col_ptr_data[fid + 1]; i++) {
            cut_fid_data[i] = fid;
        }
    }
}


template<typename T>
void syncarray_resize_cpu(SyncArray<T> &buf_array, int new_size) {
    CHECK_GE(buf_array.size(), new_size) << "The size of the target Syncarray must greater than the new size. ";
    SyncArray<T> tmp_array(new_size);
    tmp_array.copy_from(buf_array.host_data(), new_size);
    buf_array.resize(new_size);
    buf_array.copy_from(tmp_array);
}

void unique_by_flag(SyncArray<float> &target_arr, SyncArray<int> &flags, int n_columns) {
    using namespace thrust::placeholders;

//    float max_elem = max_elements(target_arr);
    float max_elem = *thrust::max_element(thrust::host, target_arr.host_data(), target_arr.host_end());
    CHECK_LT(max_elem + n_columns * (max_elem + 1), INT_MAX) << "Max_values is too large to be transformed";
    // 1. transform data into unique ranges
    thrust::transform(thrust::host,
                      target_arr.host_data(),
                      target_arr.host_end(),
                      flags.host_data(),
                      target_arr.host_data(),
                      (_1 + _2 * (max_elem + 1)));
    // 2. sort the transformed data
    thrust::sort(thrust::host, target_arr.host_data(), target_arr.host_end(), thrust::greater<float>());
    thrust::reverse(thrust::host, flags.host_data(), flags.host_end());
    // 3. eliminate duplicates
    auto new_end = thrust::unique_by_key(thrust::host, target_arr.host_data(), target_arr.host_end(),
                                         flags.host_data());
    int new_size = new_end.first - target_arr.host_data();
    syncarray_resize_cpu(target_arr, new_size);
    syncarray_resize_cpu(flags, new_size);
    // 4. transform data back
    thrust::transform(thrust::host, target_arr.host_data(),
                      target_arr.host_end(),
                      flags.host_data(),
                      target_arr.host_data(),
                      (_1 - _2 * (max_elem + 1)));
    thrust::sort_by_key(thrust::host, flags.host_data(), flags.host_end(), target_arr.host_data());
}

// cost more memory
void HistCut::get_cut_points_fast(DataSet &dataset, int max_num_bins, int n_instances) {
//    LOG(INFO) << "Fast getting cut points...";
    if(!dataset.has_csc)
        dataset.csr_to_csc();
    int n_column = dataset.n_features();

    cut_points_val.resize(dataset.csc_val.size());
    cut_col_ptr.resize(dataset.csc_col_ptr.size());
    cut_fid.resize(dataset.csc_val.size());
    cut_points_val.copy_from(&dataset.csc_val[0], dataset.csc_val.size());
    auto csc_ptr = &dataset.csc_col_ptr[0];

    auto cut_fid_data = cut_fid.host_data();
    #pragma omp parallel for
    for(int fid = 0; fid < n_column; fid ++)
        for(int i = csc_ptr[fid]; i < csc_ptr[fid+1]; i++) {
            cut_fid_data[i] = fid;
        }
    unique_by_flag(cut_points_val, cut_fid, n_column);
    //need to reassign the host_data since cut_fid is resized
    cut_fid_data = cut_fid.host_data();
    cut_col_ptr.resize(n_column + 1);
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    for(int fid = 0; fid < cut_fid.size(); fid++){
        *(cut_col_ptr_data + cut_fid_data[fid] + 1) += 1;
    }
    thrust::inclusive_scan(thrust::host, cut_col_ptr_data, cut_col_ptr_data + cut_col_ptr.size(), cut_col_ptr_data);
    SyncArray<int> select_index(cut_fid.size());
    auto select_index_data = select_index.host_data();
    #pragma omp parallel for
    for(int fid = 0; fid < n_column; fid++){
        int interval = (cut_col_ptr_data[fid+1] - cut_col_ptr_data[fid])/max_num_bins;
        for (int i = cut_col_ptr_data[fid]; i < cut_col_ptr_data[fid+1]; i++){
            int feature_idx = i - cut_col_ptr_data[fid];
            if(interval == 0)
                select_index_data[i] = 1;
            else if(feature_idx < max_num_bins)
                select_index_data[cut_col_ptr_data[fid] + interval * feature_idx] = 1;
        }
    }
    auto cut_fid_new_end = thrust::remove_if(thrust::host, cut_fid_data, cut_fid_data+cut_fid.size(), select_index_data,
                                             thrust::not1(thrust::identity<int>()));
    syncarray_resize_cpu(cut_fid, cut_fid_new_end - cut_fid_data);
    auto cut_points_val_data = cut_points_val.host_data();
    auto cut_points_val_new_end = thrust::remove_if(thrust::host, cut_points_val_data, cut_points_val.host_end(),
                                                    select_index_data, thrust::not1(thrust::identity<int>()));
    syncarray_resize_cpu(cut_points_val, cut_points_val_new_end - cut_points_val_data);

    cut_fid_data = cut_fid.host_data();
    cut_col_ptr.resize(n_column + 1);
    cut_col_ptr_data = cut_col_ptr.host_data();
    for(int fid = 0; fid < cut_fid.size(); fid++){
        *(cut_col_ptr_data + cut_fid_data[fid] + 1) += 1;
    }
    thrust::inclusive_scan(thrust::host, cut_col_ptr_data, cut_col_ptr_data + cut_col_ptr.size(), cut_col_ptr_data);

    LOG(DEBUG) << "--->>>>  cut points value: " << cut_points_val;
    LOG(DEBUG) << "--->>>> cut row ptr: " << cut_col_ptr;
    LOG(DEBUG) << "--->>>> cut fid: " << cut_fid;
    LOG(DEBUG) << "TOTAL CP:" << cut_fid.size();
    LOG(DEBUG) << "NNZ: " << dataset.csc_val.size();
}

/**
 * Generates cut points for each feature based on numeric ranges of feature values
 * @param f_range Min and max values for each feature.
 * @param max_num_bins Number of cut points for each feature.
 */
void HistCut::get_cut_points_by_feature_range(vector<vector<float>> f_range, int max_num_bins) {
    int n_features = f_range.size();

    cut_points_val.resize(n_features * max_num_bins);
    cut_col_ptr.resize(n_features + 1);
    cut_fid.resize(n_features * max_num_bins);

    auto cut_points_val_data = cut_points_val.host_data();
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    auto cut_fid_data = cut_fid.host_data();
    #pragma omp parallel for
    for(int fid = 0; fid < n_features; fid ++) {
        cut_col_ptr_data[fid] = fid * max_num_bins;
        float val_range = f_range[fid][1] - f_range[fid][0];
        float val_step = val_range / max_num_bins;
        //todo: compress the cut points if distance is small
        for(int i = 0; i < max_num_bins; i ++) {
            cut_fid_data[fid * max_num_bins + i] = fid;
            cut_points_val_data[fid * max_num_bins + i] = f_range[fid][1] - i * val_step;
        }
    }
    cut_col_ptr_data[n_features] = n_features * max_num_bins;
}



void RobustHistCut::get_cut_points_by_feature_range_balanced(DataSet &dataset, int max_bin_size, int n_instances) {
    if(!dataset.has_csc)
        dataset.csr_to_csc();
    size_t n_features = dataset.n_features();

    // obtain min-max value of each feature
    vector<vector<float_type>> f_range(n_features, vector<float_type>(2));
#pragma omp parallel for
    for (int fid = 0; fid < n_features; ++fid) {
        auto min_value = *std::min_element(dataset.csc_val.begin() + dataset.csc_col_ptr[fid],
                                  dataset.csc_val.begin() + dataset.csc_col_ptr[fid + 1]);
        auto max_value = *std::max_element(dataset.csc_val.begin() + dataset.csc_col_ptr[fid],
                                            dataset.csc_val.begin() + dataset.csc_col_ptr[fid + 1]);
        f_range[fid][0] = min_value;
        f_range[fid][1] = max_value;
    }

    float_type tol = 1e-8;
    // 2d-version of cut info for parallel
    vector<vector<float_type>> cut_points_val_vec(n_features);
    vector<vector<int>> cut_fid_vec(n_features);
    vector<int> cut_col_ptr_base(n_features + 1);
    n_instances_in_hist.resize(n_features);
#pragma omp parallel for
    for(int fid = 0; fid < n_features; fid ++) {
        float_type min_value = f_range[fid][0];
        float_type max_value = f_range[fid][1];
        std::vector<float_type> split_values  = {min_value, max_value + 0.5f};
        std::vector<std::pair<int, bool>> n_instances_in_bins_with_flag = {{n_instances, true}};

        while(true) {
            auto split_bin_id = std::distance(n_instances_in_bins_with_flag.begin(),
                                              std::max_element(n_instances_in_bins_with_flag.begin(), n_instances_in_bins_with_flag.end(),
                                                               [](std::pair<int, bool> a, std::pair<int, bool> b){
                                                                   if (!a.second) return false;
                                                                   if (!b.second) return true;
                                                                   return a.first < b.first;
                                              }));
            if (!n_instances_in_bins_with_flag[split_bin_id].second || n_instances_in_bins_with_flag[split_bin_id].first < max_bin_size)
                break;
            float_type mid_value = (split_values[split_bin_id] + split_values[split_bin_id + 1]) / 2;

            size_t n_instances_left = std::count_if(dataset.csc_val.begin() + dataset.csc_col_ptr[fid],
                                                    dataset.csc_val.begin() + dataset.csc_col_ptr[fid + 1],
                                                    [&](float_type value){
                                                        return split_values[split_bin_id] <= value && value < mid_value;
            });
            size_t n_instances_right = std::count_if(dataset.csc_val.begin() + dataset.csc_col_ptr[fid],
                                                    dataset.csc_val.begin() + dataset.csc_col_ptr[fid + 1],
                                                    [&](float_type value){
                                                        return mid_value <= value && value < split_values[split_bin_id + 1];
                                                    });
            bool left_splittable = std::abs(split_values[split_bin_id] - mid_value) > tol;
            bool right_splittable = std::abs(split_values[split_bin_id+1] - mid_value) > tol;
            split_values.insert(split_values.begin() + split_bin_id + 1, mid_value);
            n_instances_in_bins_with_flag[split_bin_id] = std::make_pair(static_cast<int>(n_instances_left), left_splittable);
            n_instances_in_bins_with_flag.insert(n_instances_in_bins_with_flag.begin() + split_bin_id + 1,
                                                 std::make_pair(static_cast<int>(n_instances_right), right_splittable));
        }

        // filter non-empty bins; remove the last split point (max)
//        cut_points_val_vec[fid].push_back(split_values[0]);
        for (int i = n_instances_in_bins_with_flag.size() - 1; i >= 0; --i) {
            if (n_instances_in_bins_with_flag[i].first > 0) {
                n_instances_in_hist[fid].push_back(n_instances_in_bins_with_flag[i].first);
                cut_points_val_vec[fid].push_back(split_values[i + 1]);
            }
        }
        cut_fid_vec[fid] = std::vector<int>(n_instances_in_hist[fid].size(), fid);
        cut_col_ptr_base[fid + 1] = static_cast<int>(n_instances_in_hist[fid].size());
    }

    cut_points_val = flatten(cut_points_val_vec);
    cut_fid = flatten(cut_fid_vec);
    cut_col_ptr.resize(dataset.n_features() + 1);
    std::inclusive_scan(cut_col_ptr_base.begin(), cut_col_ptr_base.end(), cut_col_ptr.begin());
//    LOG(INFO);
}

void RobustHistCut::get_cut_points_by_instance(DataSet &dataset, int max_num_bins, int n_instances) {
    if(!dataset.has_csc)
        dataset.csr_to_csc();
    int n_column = dataset.n_features();

    SyncArray<float_type> cut_points_val(dataset.csc_val.size());
    SyncArray<int> cut_col_ptr(dataset.csc_col_ptr.size());
    SyncArray<int> cut_fid(dataset.csc_val.size());

    cut_points_val.copy_from(&dataset.csc_val[0], dataset.csc_val.size());
    auto csc_ptr = &dataset.csc_col_ptr[0];

    auto cut_fid_data = cut_fid.host_data();
#pragma omp parallel for
    for(int fid = 0; fid < n_column; fid ++)
        for(int i = csc_ptr[fid]; i < csc_ptr[fid+1]; i++) {
            cut_fid_data[i] = fid;
        }
    unique_by_flag(cut_points_val, cut_fid, n_column);
    //need to reassign the host_data since cut_fid is resized
    cut_fid_data = cut_fid.host_data();
    cut_col_ptr.resize(n_column + 1);
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    for(int fid = 0; fid < cut_fid.size(); fid++){
        *(cut_col_ptr_data + cut_fid_data[fid] + 1) += 1;
    }
    thrust::inclusive_scan(thrust::host, cut_col_ptr_data, cut_col_ptr_data + cut_col_ptr.size(), cut_col_ptr_data);
    SyncArray<int> select_index(cut_fid.size());
    auto select_index_data = select_index.host_data();
#pragma omp parallel for
    for(int fid = 0; fid < n_column; fid++){
        int interval = (cut_col_ptr_data[fid+1] - cut_col_ptr_data[fid])/max_num_bins;
        for (int i = cut_col_ptr_data[fid]; i < cut_col_ptr_data[fid+1]; i++){
            int feature_idx = i - cut_col_ptr_data[fid];
            if(interval == 0)
                select_index_data[i] = 1;
            else if(feature_idx < max_num_bins)
                select_index_data[cut_col_ptr_data[fid] + interval * feature_idx] = 1;
        }
    }
    auto cut_fid_new_end = thrust::remove_if(thrust::host, cut_fid_data, cut_fid_data+cut_fid.size(), select_index_data,
                                             thrust::not1(thrust::identity<int>()));
    syncarray_resize_cpu(cut_fid, cut_fid_new_end - cut_fid_data);
    auto cut_points_val_data = cut_points_val.host_data();
    auto cut_points_val_new_end = thrust::remove_if(thrust::host, cut_points_val_data, cut_points_val.host_end(),
                                                    select_index_data, thrust::not1(thrust::identity<int>()));
    syncarray_resize_cpu(cut_points_val, cut_points_val_new_end - cut_points_val_data);

    cut_fid_data = cut_fid.host_data();
    cut_col_ptr.resize(n_column + 1);
    cut_col_ptr_data = cut_col_ptr.host_data();
    for(int fid = 0; fid < cut_fid.size(); fid++){
        *(cut_col_ptr_data + cut_fid_data[fid] + 1) += 1;
    }
    thrust::inclusive_scan(thrust::host, cut_col_ptr_data, cut_col_ptr_data + cut_col_ptr.size(), cut_col_ptr_data);

    this->cut_points_val = cut_points_val.to_vec();
    this->cut_col_ptr = cut_col_ptr.to_vec();
    this->cut_fid = cut_fid.to_vec();

    LOG(DEBUG) << "--->>>>  cut points value: " << cut_points_val;
    LOG(DEBUG) << "--->>>> cut row ptr: " << cut_col_ptr;
    LOG(DEBUG) << "--->>>> cut fid: " << cut_fid;
    LOG(DEBUG) << "TOTAL CP:" << cut_fid.size();
    LOG(DEBUG) << "NNZ: " << dataset.csc_val.size();
}
