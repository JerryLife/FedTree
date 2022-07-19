//
// Created by HUSTW on 7/31/2021.
//

#include "FedTree/Tree/deltaboost.h"
#include "FedTree/booster.h"
#include "FedTree/deltabooster.h"
#include "FedTree/Tree/delta_tree_remover.h"
#include "FedTree/Tree/deltaboost_remover.h"

void DeltaBoost::train(DeltaBoostParam &param, DataSet &dataset) {
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist") {
        std::cout << "FedTree only supports histogram-based training yet";
        exit(1);
    }

    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if (param.num_class > 2)
            param.tree_per_rounds = param.num_class;
    } else if (param.objective.find("reg:") != std::string::npos) {
        param.num_class = 1;
    }

//    std::map<int, vector<int>> batch_idxs;
//    Partition partition;
//    vector<DataSet> subsets(3);
//    partition.homo_partition(dataset, 3, true, subsets, batch_idxs);
//
    LOG(INFO) << "starting building trees";

    DeltaBooster booster;
    booster.init(dataset, param);
    dataset.set_seed(0);
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    for (int i = 0; i < param.n_trees; ++i) {
        // subsampling by hashing
        if (param.hash_sampling_round > 1) {
            if (i % param.hash_sampling_round == 0) {
                dataset.update_sampling_by_hashing_(param.hash_sampling_round);
            }
            auto &sub_dataset = dataset.get_sampled_dataset(i % param.hash_sampling_round);
            booster.reset(sub_dataset, param);
            if (i > 0) {
                predict_raw(param, sub_dataset, booster.fbuilder->y_predict);
            }

            auto subset_indices = dataset.get_subset_indices(i % param.hash_sampling_round);
            is_subset_indices_in_tree.emplace_back(indices_to_hash_table(subset_indices, dataset.n_instances()));
        }
        //one iteration may produce multiple trees, depending on objectives
        booster.boost(trees, gh_pairs_per_sample, ins2node_indices_per_tree);
        int valid_size = 0;
        for (const auto &node: trees[trees.size() - 1][0].nodes) {
            if (node.is_valid) ++valid_size;
        }
        LOG(INFO) << "Tree " << i << ", Number of nodes:" << valid_size;
    }

//    float_type score = predict_score(param, dataset);
//    LOG(INFO) << score;

    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count();

    return;
}


void DeltaBoost::remove_samples(DeltaBoostParam &param, DataSet &dataset, const vector<int>& sample_indices) {


    SyncArray<float_type> y = SyncArray<float_type>(dataset.n_instances());
    y.copy_from(dataset.y.data(), dataset.n_instances());
    std::unique_ptr<ObjectiveFunction> obj(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataset);     // slicing param

    LOG(INFO) << "Preparing for deletion";

    std::vector<std::vector<DeltaTree>> used_trees(trees.begin(), trees.begin() + param.n_used_trees);

    DeltaBoostRemover deltaboost_remover;
    if (param.hash_sampling_round > 1) {
        deltaboost_remover = DeltaBoostRemover(&dataset, &trees, is_subset_indices_in_tree, obj.get(), param);
    } else {
        typedef std::chrono::high_resolution_clock clock;
        auto start_time = clock::now();

        deltaboost_remover = DeltaBoostRemover(&dataset, &trees, obj.get(), param);

        auto end_time = clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        LOG(DEBUG) << "[Removing time] Step 0 (out) = " << duration.count();
    }



    deltaboost_remover.get_info_by_prediction();
    LOG(INFO) << "Deleting...";

    for (int i = 0; i < used_trees.size(); ++i) {
//        DeltaTree &tree = trees[i][0];
//        vector<GHPair>& gh_pairs = gh_pairs_per_sample[i];
//        auto &ins2node_indices = ins2node_indices_per_tree[i];
//        DeltaTreeRemover tree_remover(&tree, &dataset, param, gh_pairs, ins2node_indices);

        DeltaTreeRemover& tree_remover = deltaboost_remover.tree_removers[i];
        vector<bool> is_iid_removed = indices_to_hash_table(sample_indices, dataset.n_instances());
        tree_remover.is_iid_removed = is_iid_removed;
        const std::vector<GHPair>& gh_pairs = tree_remover.gh_pairs;
        vector<int> trained_sample_indices;
        if (param.hash_sampling_round > 1) {
            std::copy_if(sample_indices.begin(), sample_indices.end(), std::back_inserter(trained_sample_indices), [&](int idx){
                return is_subset_indices_in_tree[i][idx];
            });
        } else {
            trained_sample_indices = sample_indices;
        }

        tree_remover.remove_samples_by_indices(trained_sample_indices);

        if (i > 0) {
            SyncArray<float_type> y_predict;
            predict_raw(param, dataset, y_predict, i);

            SyncArray<GHPair> updated_gh_pairs_array(y.size());
            obj->get_gradient(y, y_predict, updated_gh_pairs_array);
            vector<GHPair> delta_gh_pairs = updated_gh_pairs_array.to_vec();
            GHPair sum_gh_pair = std::accumulate(delta_gh_pairs.begin(), delta_gh_pairs.end(), GHPair());

            vector<int> adjust_indices;
            vector<GHPair> adjust_values;
            for (int j = 0; j < delta_gh_pairs.size(); ++j) {
                if (is_iid_removed[j] || (param.hash_sampling_round > 1 && !is_subset_indices_in_tree[i][j])) continue;
                if (std::fabs(delta_gh_pairs[j].g - gh_pairs[j].g) > 1e-6 ||
                    std::fabs(delta_gh_pairs[j].h - gh_pairs[j].h) > 1e-6) {
                    adjust_indices.emplace_back(j);
                    adjust_values.emplace_back(delta_gh_pairs[j] - gh_pairs[j]);
                }
            }
//            GHPair sum_delta_gh_pair = std::accumulate(adjust_values.begin(), adjust_values.end(), GHPair());

//            // debug only
//            SyncArray<int> adjust_indices_array;
//            SyncArray<GHPair> adjust_values_array;
//            adjust_indices_array.load_from_vec(adjust_indices);
//            adjust_values_array.load_from_vec(adjust_values);
//            LOG(DEBUG) << "Adjusted indices" << adjust_indices_array;
//            LOG(DEBUG) << "Adjusted values" << adjust_values_array;

            tree_remover.adjust_split_nbrs_by_indices(adjust_indices, adjust_values, false);
        }
    }
}

float_type DeltaBoost::predict_score(const DeltaBoostParam &model_param, const DataSet &dataSet, int num_trees) {
    SyncArray<float_type> y_predict;
    predict_raw(model_param, dataSet, y_predict, num_trees);
    LOG(DEBUG) << "y_predict:" << y_predict;
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    float_type score = metric->get_score(y_predict);

//    LOG(INFO) << metric->get_name().c_str() << " = " << score;
    LOG(INFO) << "Test: " << metric->get_name() << " = " << score;
    return score;
}

void DeltaBoost::predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict,
                             int num_trees) {
    TIMED_SCOPE(timerObj, "predict");
    int n_instances = dataSet.n_instances();
//    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = num_trees == -1 ? trees.size() : num_trees;
    int num_class = trees.front().size();

//    int total_num_node = num_iter * num_class * num_node;
    //TODO: reduce the output size for binary classification
    y_predict.resize(n_instances * num_class);

//    vector<DeltaTree::DeltaNode> model(total_num_node, DeltaTree::DeltaNode());
//    int tree_cnt = 0;
//    for (auto &vtree:trees) {
//        for (auto &t:vtree) {
//            memcpy(model.data() + num_node * tree_cnt, t.nodes.data(), sizeof(DeltaTree::DeltaNode) * num_node);
//            tree_cnt++;
//        }
//    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");
    //copy instances from to GPU
    SyncArray<int> csr_col_idx(dataSet.csr_col_idx.size());
    SyncArray<float_type> csr_val(dataSet.csr_val.size());
    SyncArray<int> csr_row_ptr(dataSet.csr_row_ptr.size());
    csr_col_idx.copy_from(dataSet.csr_col_idx.data(), dataSet.csr_col_idx.size());
    csr_val.copy_from(dataSet.csr_val.data(), dataSet.csr_val.size());
    csr_row_ptr.copy_from(dataSet.csr_row_ptr.data(), dataSet.csr_row_ptr.size());

    //do prediction
//    auto model_host_data = model.data();
    auto predict_data = y_predict.host_data();
    auto csr_col_idx_data = csr_col_idx.host_data();
    auto csr_val_data = csr_val.host_data();
    auto csr_row_ptr_data = csr_row_ptr.host_data();
    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

    //predict BLOCK_SIZE instances in a block, 1 thread for 1 instance
//    int BLOCK_SIZE = 128;
    //determine whether we can use shared memory
//    size_t smem_size = n_features * BLOCK_SIZE * sizeof(float_type);
//    int NUM_BLOCK = (n_instances - 1) / BLOCK_SIZE + 1;

    //use sparse format and binary search
#pragma omp parallel for      // remove for debug
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](const DeltaTree::DeltaNode& node, float_type feaValue) {
            return feaValue < node.split_value ? node.lch_index : node.rch_index;
//            return ft_ge(feaValue, node.split_value) ? node.rch_index : node.lch_index;
        };
        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };
        int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
//                const DeltaTree::DeltaNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
//                DeltaTree::DeltaNode curNode = node_data[0];
                DeltaTree::DeltaNode cur_node = trees[iter][t].nodes[0];
                const DeltaTree::DeltaNode* node_data = trees[iter][t].nodes.data();
                int cur_nid = 0; //node id
                int depth = 0;
                int last_idx = -1;
                while (!cur_node.is_leaf) {
                    if (cur_node.lch_index < 0 || cur_node.rch_index < 0) {
                        LOG(FATAL);
                    }
                    int fid = cur_node.split_feature_id;
                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    last_idx = cur_node.final_id;
                    if (!is_missing)
                        cur_nid = get_next_child(cur_node, fval);
                    else if (cur_node.default_right)
                        cur_nid = cur_node.rch_index;
                    else
                        cur_nid = cur_node.lch_index;
                    const auto& cur_potential_node = node_data[cur_nid];
                    int cur_node_idx = cur_potential_node.potential_nodes_indices[0];
                    cur_node = node_data[cur_node_idx];
                    depth++;
                }
                sum += lr * node_data[cur_nid].base_weight;
            }
            predict_data_class[iid] += sum;
        }//end all tree prediction
    }
}

vector<float_type> DeltaBoost::predict_raw(const DeltaBoostParam &model_param, const DataSet &dataSet, int num_trees) {
    /**
     * This function is a wrapper for predict_raw with SyncArray. The return value is expected to be used for initialization
     * instead of assignment. E.g., auto y_predict = predict_raw(param, ...); In this way, the copying would be optimized
     * by "named return value optimization (NRVO)".
     */
    SyncArray<float_type> y_predict;
    predict_raw(model_param, dataSet, y_predict, num_trees);
    return y_predict.to_vec();
}

