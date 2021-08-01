//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/Tree/delta_tree_builder.h"
#include "thrust/sequence.h"

void DeltaTreeBuilder::init(DataSet &dataset, const DeltaBoostParam &param) {
    HistTreeBuilder::init(dataset, param);
    this->param = param;
}

void DeltaTreeBuilder::init_nocutpoints(DataSet &dataset, const DeltaBoostParam &param) {
    HistTreeBuilder::init(dataset, param);
    this->param = param;
}

vector<DeltaTree> DeltaTreeBuilder::build_delta_approximate(const SyncArray<GHPair> &gradients, bool update_y_predict) {
    vector<DeltaTree> trees(param.tree_per_rounds);
    TIMED_FUNC(timerObj);

    for (int k = 0; k < param.tree_per_rounds; ++k) {
        DeltaTree &tree = trees[k];

        this->ins2node_id.resize(n_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
        this->tree.init_CPU(this->gradients, param);

        for (int level = 0; level < param.depth; ++level) {
            //LOG(INFO)<<"in level:"<<level;
            find_split(level);
            //split_point_all_reduce(level);
            {
                TIMED_SCOPE(timerObj, "apply sp");
                update_tree();
                //  LOG(INFO) << this->trees.nodes;
                update_ins2node_id();
//                LOG(INFO) << "ins2node_id: " << ins2node_id;
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    if (!has_split) {
                        LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
                //ins2node_id_all_reduce(level);
            }
        }
        //here
        this->tree.prune_self(param.gamma);
//        LOG(INFO) << "y_predict: " << y_predict;
        if(update_y_predict)
            predict_in_training(k);
        tree.nodes.resize(this->tree.nodes.size());
        tree.nodes.copy_from(this->tree.nodes);
    }
    return trees;
}

void DeltaTreeBuilder::find_split(int level) {
    TIMED_FUNC(timerObj);
    std::chrono::high_resolution_clock timer;
    int n_nodes_in_level = 1 << level;
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = sorted_dataset.n_features();
    int n_partition = n_column * n_nodes_in_level;
    int n_bins = cut.cut_points_val.size();
    int n_max_nodes = 2 << param.depth;
    int n_max_splits = n_max_nodes * n_bins;

    auto cut_fid_data = cut.cut_fid.host_data();

//    auto i2fid = [=] __host__(int i) { return cut_fid_data[i % n_bins]; };
//    auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);

    SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
    auto hist_fid_data = hist_fid.host_data();

#pragma omp parallel for
    for (int i = 0; i < hist_fid.size(); i++)
        hist_fid_data[i] = cut_fid_data[i % n_bins];


    int n_split = n_nodes_in_level * n_bins;
    SyncArray<GHPair> missing_gh(n_partition);
    LOG(TRACE) << "start finding split";

    auto t_build_start = timer.now();

    SyncArray<GHPair> hist(n_max_splits);
    SyncArray<float_type> gain(n_max_splits);
    compute_histogram_in_a_level(level, n_max_splits, n_bins, n_nodes_in_level, hist_fid_data, missing_gh, hist);
    //LOG(INFO) << hist;
    compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data, missing_gh, hist);
    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
    get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
    //LOG(INFO) << best_idx_gain;
    get_split_points(best_idx_gain, n_nodes_in_level, hist_fid_data, missing_gh, hist);
    //LOG(INFO) << this->sp;
}

//todo: reduce hist size according to current level (not n_max_split)
void DeltaTreeBuilder::compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                                   int *hist_fid, SyncArray<GHPair> &missing_gh,
                                                   SyncArray<GHPair> &hist) {
    std::chrono::high_resolution_clock timer;

    SyncArray<int> &nid = ins2node_id;
    SyncArray<GHPair> &gh_pair = gradients;
    DeltaTree &tree = this->tree;
    SyncArray<SplitPoint> &sp = this->sp;
    HistCut &cut = this->cut;
    auto &dense_bin_id = this->dense_bin_id;
    auto &last_hist = this->last_hist;

    TIMED_FUNC(timerObj);
//    int n_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = sorted_dataset.n_features();
    int n_partition = n_column * n_nodes_in_level;
//    int n_bins = cut.cut_points_val.size();
//    int n_max_nodes = 2 << param.depth;
//    int n_max_splits = n_max_nodes * n_bins;
    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";

    {
        TIMED_SCOPE(timerObj, "build hist");
        if (n_nodes_in_level == 1) {
            auto hist_data = hist.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;
            auto n_instances = this->n_instances;
//                ThunderGBM: check size of histogram.
            //has bug if using openmp
//            #pragma omp parallel for
            for (int i = 0; i < n_instances * n_column; i++) {
                int iid = i / n_column;
                int fid = i % n_column;
                unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                if (bid != max_num_bin) {
                    int feature_offset = cut_col_ptr_data[fid];
                    const GHPair src = gh_data[iid];
                    GHPair &dest = hist_data[feature_offset + bid];
                    dest = dest + src;
//                    g and h values are 0 if after HE encryption
//                    if (src.h != 0) {
////                        #pragma omp atomic
//                        dest.h += src.h;
//                    }
//                    if (src.g != 0) {
////                        #pragma omp atomic
//                        dest.g += src.g;
//                    }

                }
            }
        } else {
            auto t_dp_begin = timer.now();
            SyncArray<int> node_idx(n_instances);
            SyncArray<int> node_ptr(n_nodes_in_level + 1);
            {
                TIMED_SCOPE(timerObj, "data partitioning");
                SyncArray<int> nid4sort(n_instances);
                nid4sort.copy_from(ins2node_id);
                sequence(thrust::host, node_idx.host_data(), node_idx.host_end(), 0);
                thrust::stable_sort_by_key(thrust::host, nid4sort.host_data(), nid4sort.host_end(),
                                           node_idx.host_data());
                auto counting_iter = thrust::make_counting_iterator<int>(nid_offset);
                node_ptr.host_data()[0] =
                        thrust::lower_bound(thrust::host, nid4sort.host_data(), nid4sort.host_end(), nid_offset) -
                        nid4sort.host_data();

                thrust::upper_bound(thrust::host, nid4sort.host_data(), nid4sort.host_end(), counting_iter,
                                    counting_iter + n_nodes_in_level, node_ptr.host_data() + 1);
            }
            auto t_dp_end = timer.now();
            std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
            this->total_dp_time += dp_used_time.count();


            auto node_ptr_data = node_ptr.host_data();
            auto node_idx_data = node_idx.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;

            for (int i = 0; i < n_nodes_in_level / 2; ++i) {

                int nid0_to_compute = i * 2;
                int nid0_to_substract = i * 2 + 1;
                //node_ptr_data[i+1] - node_ptr_data[i] is the number of instances in node i, i is the node id in current level (start from 0)
                int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
                int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                if (std::max(n_ins_left, n_ins_right) == 0) continue;
                //only compute the histogram on the node with the smaller data
                if (n_ins_left > n_ins_right)
                    std::swap(nid0_to_compute, nid0_to_substract);
                //compute histogram
                {
                    int nid0 = nid0_to_compute;
                    auto idx_begin = node_ptr.host_data()[nid0];
                    auto idx_end = node_ptr.host_data()[nid0 + 1];
                    auto hist_data = hist.host_data() + nid0 * n_bins;
                    this->total_hist_num++;
                    //                ThunderGBM: check size of histogram.
                    //has bug if using openmp
//#pragma omp parallel for
                    for (int i = 0; i < (idx_end - idx_begin) * n_column; i++) {

                        int iid = node_idx_data[i / n_column + idx_begin];
                        int fid = i % n_column;
                        unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                        if (bid != max_num_bin) {
                            int feature_offset = cut_col_ptr_data[fid];
                            const GHPair src = gh_data[iid];
                            GHPair &dest = hist_data[feature_offset + bid];
//                            if (src.h != 0) {
////                                #pragma omp atomic
//                                dest.h += src.h;
//                            }
//                            if (src.g != 0) {
////                                #pragma omp atomic
//                                dest.g += src.g;
//                            }
                            dest = dest + src;
                        }
                    }
                }

                //subtract to the histogram of the other node
                auto t_copy_start = timer.now();
                {
                    auto hist_data_computed = hist.host_data() + nid0_to_compute * n_bins;
                    auto hist_data_to_compute = hist.host_data() + nid0_to_substract * n_bins;
                    auto father_hist_data = last_hist.host_data() + (nid0_to_substract / 2) * n_bins;
//#pragma omp parallel for
                    for (int i = 0; i < n_bins; i++) {
                        hist_data_to_compute[i] = father_hist_data[i] - hist_data_computed[i];
                    }
                }
                auto t_copy_end = timer.now();
                std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
                this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);
            }  // end for each node
        }
        last_hist.resize(n_nodes_in_level * n_bins);
        auto last_hist_data = last_hist.host_data();
        auto hist_data = hist.host_data();
        for (int i = 0; i < n_nodes_in_level * n_bins; i++) {
            last_hist_data[i] = hist_data[i];
        }
    }

    this->build_n_hist++;
    inclusive_scan_by_key(thrust::host, hist_fid, hist_fid + n_split,
                          hist.host_data(), hist.host_data());
    LOG(DEBUG) << hist;

    auto nodes_data = tree.nodes.host_data();
    auto missing_gh_data = missing_gh.host_data();
    auto cut_col_ptr = cut.cut_col_ptr.host_data();
    auto hist_data = hist.host_data();
//#pragma omp parallel for
    for (int pid = 0; pid < n_partition; pid++) {
        int nid0 = pid / n_column;
        int nid = nid0 + nid_offset;
        //            todo: check, ThunderGBM uses return;
        if (!nodes_data[nid].splittable()) continue;
        int fid = pid % n_column;
        if (cut_col_ptr[fid + 1] != cut_col_ptr[fid]) {
            GHPair node_gh = hist_data[nid0 * n_bins + cut_col_ptr[fid + 1] - 1];
            missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
        }
    }
    return;
}


void DeltaTreeBuilder::update_ins2node_id() {
    TIMED_FUNC(timerObj);
    SyncArray<bool> has_splittable(1);
//    auto &columns = shards.columns;
    //set new node id for each instance
    {
//        TIMED_SCOPE(timerObj, "get new node id");
        auto nid_data = ins2node_id.host_data();
        DeltaTree::DeltaNode *nodes_data = tree.nodes.host_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.host_data();
        int column_offset = 0;

        int n_column = sorted_dataset.n_features();
        auto dense_bin_id_data = dense_bin_id.host_data();
        int max_num_bin = param.max_num_bin;
#pragma omp parallel for
        for (int iid = 0; iid < n_instances; iid++) {
            int nid = nid_data[iid];
            const DeltaTree::DeltaNode &node = nodes_data[nid];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                h_s_data[0] = true;
                unsigned char split_bid = node.split_bid;
                unsigned char bid = dense_bin_id_data[iid * n_column + split_fid - column_offset];
                bool to_left = true;
                if ((bid == max_num_bin && node.default_right) || (bid <= split_bid))
                    to_left = false;
                if (to_left) {
                    //goes to left child
                    nid_data[iid] = node.lch_index;
//                    #pragma omp atomic
                    nodes_data[node.lch_index].n_instances += 1;
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
//                    #pragma omp atomic
                    nodes_data[node.rch_index].n_instances += 1;
                }
            }
        }
    }
    LOG(DEBUG) << "new tree_id = " << ins2node_id;
    has_split = has_splittable.host_data()[0];
}


void DeltaTreeBuilder::update_tree() {
    TIMED_FUNC(timerObj);
    auto& sp = this->sp;
    auto& tree = this->tree;
    auto sp_data = sp.host_data();
    int n_nodes_in_level = sp.size();

    DeltaTree::DeltaNode *nodes_data = tree.nodes.host_data();
    float_type rt_eps = param.rt_eps;
    float_type lambda = param.lambda;

#pragma omp parallel for
    for(int i = 0; i < n_nodes_in_level; i++){
        float_type best_split_gain = sp_data[i].gain;
        if (best_split_gain > rt_eps) {
            //do split
            //todo: check, thundergbm uses return
            if (sp_data[i].nid == -1) continue;
            int nid = sp_data[i].nid;
            DeltaTree::DeltaNode &node = nodes_data[nid];
            node.gain = best_split_gain;

            DeltaTree::DeltaNode &lch = nodes_data[node.lch_index];//left child
            DeltaTree::DeltaNode &rch = nodes_data[node.rch_index];//right child
            lch.is_valid = true; //TODO: broadcast lch and rch
            rch.is_valid = true;
            node.split_feature_id = sp_data[i].split_fea_id;
            GHPair p_missing_gh = sp_data[i].fea_missing_gh;
            //todo process begin
            node.split_value = sp_data[i].fval;
            node.split_bid = sp_data[i].split_bid;
            rch.sum_gh_pair = sp_data[i].rch_sum_gh;
            if (sp_data[i].default_right) {
                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
                // LOG(INFO) << "RCH" << rch.sum_gh_pair;
                node.default_right = true;
            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
            //  LOG(INFO) << "LCH" << lch.sum_gh_pair;
            lch.calc_weight(lambda);
            rch.calc_weight(lambda);
        } else {
            //set leaf
            //todo: check, thundergbm uses return
            if (sp_data[i].nid == -1) continue;
            int nid = sp_data[i].nid;
            DeltaTree::DeltaNode &node = nodes_data[nid];
            node.is_leaf = true;
            nodes_data[node.lch_index].is_valid = false;
            nodes_data[node.rch_index].is_valid = false;
        }
    }
    // LOG(INFO) << tree.nodes;
}


void DeltaTreeBuilder::predict_in_training(int k) {
    auto y_predict_data = y_predict.host_data() + k * n_instances;
    auto nid_data = ins2node_id.host_data();
    const DeltaTree::DeltaNode *nodes_data = tree.nodes.host_data();
    auto lr = param.learning_rate;
#pragma omp parallel for
    for(int i = 0; i < n_instances; i++){
        int nid = nid_data[i];
        while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
        y_predict_data[i] += lr * nodes_data[nid].base_weight;
    }
}


void
DeltaTreeBuilder::compute_gain_in_a_level(SyncArray<float_type> &gain, int n_nodes_in_level, int n_bins, int *hist_fid,
                                         SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist, int n_column) {
//    SyncArray<compute_gainfloat_type> gain(n_max_splits);
    if (n_column == 0)
        n_column = sorted_dataset.n_features();
    int n_split = n_nodes_in_level * n_bins;
    int nid_offset = static_cast<int>(n_nodes_in_level - 1);
    auto compute_gain = []__host__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                   float_type lambda) -> float_type {
        if (lch.h >= min_child_weight && rch.h >= min_child_weight)
            return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
                   (father.g * father.g) / (father.h + lambda);
        else
            return 0;
    };
    const DeltaTree::DeltaNode * nodes_data = tree.nodes.host_data();
    GHPair *gh_prefix_sum_data = hist.host_data();
    float_type *gain_data = gain.host_data();
    const auto missing_gh_data = missing_gh.host_data();
//    auto ignored_set_data = ignored_set.host_data();
    //for lambda expression
    float_type mcw = param.min_child_weight;
    float_type l = param.lambda;

#pragma omp parallel for
    for (int i = 0; i < n_split; i++) {
        int nid0 = i / n_bins;
        int nid = nid0 + nid_offset;
        int fid = hist_fid[i % n_bins];
        if (nodes_data[nid].is_valid) {
            int pid = nid0 * n_column + fid;
            GHPair father_gh = nodes_data[nid].sum_gh_pair;
            GHPair p_missing_gh = missing_gh_data[pid];
            GHPair rch_gh = gh_prefix_sum_data[i];
            float_type default_to_left_gain = std::max(0.f,
                                                       compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
//            rch_gh = rch_gh + p_missing_gh;
            rch_gh.g += p_missing_gh.g;
            rch_gh.h += p_missing_gh.h;
            float_type default_to_right_gain = std::max(0.f,
                                                        compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            if (default_to_left_gain > default_to_right_gain)
                gain_data[i] = default_to_left_gain;
            else
                gain_data[i] = -default_to_right_gain;//negative means default split to right
        } else gain_data[i] = 0;
    }
    return;
}

void DeltaTreeBuilder::get_split_points(SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int *hist_fid,
                                       SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist) {
//    TIMED_SCOPE(timerObj, "get split points");
    int nid_offset = static_cast<int>(n_nodes_in_level - 1);
    const int_float *best_idx_gain_data = best_idx_gain.host_data();
    auto hist_data = hist.host_data();
    const auto missing_gh_data = missing_gh.host_data();
    auto cut_val_data = cut.cut_points_val.host_data();

    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    auto nodes_data = tree.nodes.host_data();

    auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++) {
        int_float bst = best_idx_gain_data[i];
        float_type best_split_gain = thrust::get < 1 > (bst);
        int split_index = thrust::get < 0 > (bst);
        if (!nodes_data[i + nid_offset].is_valid) {
            sp_data[i].split_fea_id = -1;
            sp_data[i].nid = -1;
            // todo: check, ThunderGBM uses return;
            continue;
        }
        int fid = hist_fid[split_index];
        sp_data[i].split_fea_id = fid;
        sp_data[i].nid = i + nid_offset;
        sp_data[i].gain = fabsf(best_split_gain);
        int n_bins = cut.cut_points_val.size();
        int n_column = sorted_dataset.n_features();
        sp_data[i].fval = cut_val_data[split_index % n_bins];
        sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_col_ptr_data[fid]);
        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
        sp_data[i].no_split_value_update = 0;
    }
    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}