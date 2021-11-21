//
// Created by liqinbin on 10/13/20.
//

#include "FedTree/FL/FLparam.h"
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/gbdt.h"
#include "FedTree/Tree/deltaboost.h"


#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif


int main(int argc, char** argv){
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

/*
    //initialize parameters
    FLParam fl_param;
    Parser parser;
    parser.parse_param(fl_param, argc, argv);

    //load dataset from file/files
    DataSet dataset;
    dataset.load_from_file(fl_param.dataset_path);

    //initialize parties and server *with the dataset*
    vector<Party> parties;
    for(i = 0; i < fl_param.n_parties; i++){
        Party party;
        parties.push_back(party);
    }
    Server server;

    //train
    FLtrainer trainer;
    model = trainer.train(parties, server, fl_param);

    //test
    Dataset test_dataset;
    test_dataset.load_from_file(fl_param.test_dataset_path);
    acc = model.predict(test_dataset);
*/

//centralized training test
    FLParam fl_param;
    Parser parser;
    parser.parse_param(fl_param, argc, argv);
    GBDTParam &model_param = fl_param.gbdt_param;
    if(model_param.verbose == 0) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
    }
    else if (model_param.verbose == 1) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }

    if (!model_param.profiling) {
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }
//    if(fl_param.mode == "centralized") {
//        DataSet dataset;
//        vector <vector<Tree>> boosted_model;
//        dataset.load_from_file(model_param.path, fl_param);
//        std::map<int, vector<int>> batch_idxs;
//        Partition partition;
//        vector<DataSet> subsets(3);
//        partition.homo_partition(dataset, 3, true, subsets, batch_idxs);
//        GBDT gbdt;
//        gbdt.train(model_param, dataset);
////       float_type score = gbdt.predict_score(model_param, dataset);
//       // LOG(INFO) << score;
//      //  parser.save_model("tgbm.model", model_param, gbdt.trees, dataset);
//    }
//    }
//    else{
    int n_parties = fl_param.n_parties;
    vector<DataSet> train_subsets(n_parties);
    vector<DataSet> test_subsets(n_parties);
    vector<DataSet> subsets(n_parties);
    vector<SyncArray<bool>> feature_map(n_parties);
    std::map<int, vector<int>> batch_idxs;
    DataSet dataset;
    bool use_global_test_set = !model_param.test_path.empty();
    dataset.load_from_file(model_param.path, fl_param);
    if (fl_param.partition == true && fl_param.mode != "centralized") {
        Partition partition;
        if (fl_param.partition_mode == "hybrid") {
            LOG(INFO) << "horizontal vertical dir";
            if (fl_param.mode == "horizontal")
                CHECK_EQ(fl_param.n_verti, 1);
            if (fl_param.mode == "vertical")
                CHECK_EQ(fl_param.n_hori, 1);
            partition.horizontal_vertical_dir_partition(dataset, n_parties, fl_param.alpha, feature_map, subsets,
                                                        fl_param.n_hori, fl_param.n_verti);
//            std::cout<<"subsets[0].n_instances:"<<subsets[0].n_instances()<<std::endl;
//            std::cout<<"subsets[0].nnz:"<<subsets[0].csr_val.size()<<std::endl;
//            std::cout<<"subsets[1].n_instances:"<<subsets[1].n_instances()<<std::endl;
//            std::cout<<"subsets[1].nnz:"<<subsets[1].csr_val.size()<<std::endl;
//            std::cout<<"subsets[2].n_instances:"<<subsets[2].n_instances()<<std::endl;
//            std::cout<<"subsets[2].nnz:"<<subsets[2].csr_val.size()<<std::endl;
//            std::cout<<"subsets[3].n_instances:"<<subsets[3].n_instances()<<std::endl;
//            std::cout<<"subsets[3].nnz:"<<subsets[3].csr_val.size()<<std::endl;
        } else if (fl_param.partition_mode == "vertical") {
            CHECK_EQ(fl_param.mode, "vertical");
            dataset.csr_to_csc();
            partition.homo_partition(dataset, n_parties, false, subsets, batch_idxs);
            if (!use_global_test_set) {
                LOG(INFO) << "train test split";
                for (int i = 0; i < n_parties; i++) {
                    partition.train_test_split(subsets[i], train_subsets[i], test_subsets[i]);
                }
            }else{
                    for (int i = 0; i < n_parties; i++) {
                        train_subsets[i] = subsets[i];
                    }
            }
        }else if (fl_param.partition_mode=="horizontal") {
            partition.homo_partition(dataset, n_parties, true, subsets, batch_idxs);
            if (!use_global_test_set) {
                LOG(INFO) << "train test split";
                for (int i = 0; i < n_parties; i++) {
                    partition.train_test_split(subsets[i], train_subsets[i], test_subsets[i]);
                }
            }else{
                for (int i = 0; i < n_parties; i++) {
                    train_subsets[i] = subsets[i];
                }
            }
        }
    }

    DataSet test_dataset;
    if (use_global_test_set)
        test_dataset.load_from_file(model_param.test_path, fl_param);

//    if (ObjectiveFunction::need_group_label(param.gbdt_param.objective)) {
//        group_label();
//        param.gbdt_param.num_class = label.size();
//    }

    GBDTParam &param = fl_param.gbdt_param;

//    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos || param.metric == "error") {
//        for (int i = 0; i < n_parties; i++) {
//            train_subsets[i].group_label();
//            test_subsets[i].group_label();
//        }
//        int num_class = dataset.label.size();
//        if (param.num_class != num_class) {
//            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
//            param.num_class = num_class;
//        }
//        if(param.num_class > 2)
//            param.tree_per_rounds = param.num_class;
//    }
//    else if(param.objective.find("reg:") != std::string::npos){
//        param.num_class = 1;
//    }

    vector<Party> parties(n_parties);
    vector<int> n_instances_per_party(n_parties);
    Server server;
    if(fl_param.mode != "centralized") {
        LOG(INFO) << "initialize parties";
        for (int i = 0; i < n_parties; i++) {
            parties[i].init(i, train_subsets[i], fl_param, feature_map[i]);
            n_instances_per_party[i] = train_subsets[i].n_instances();
        }
        LOG(INFO) << "initialize server";
        if (fl_param.mode == "vertical") {
            server.vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y, dataset.label);
        } else if (fl_param.mode == "horizontal") {
            server.horizontal_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset);
        } else {
            server.init(fl_param, dataset.n_instances(), n_instances_per_party);
        }
    }

    LOG(INFO) << "start training";
    FLtrainer trainer;
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist"){
        std::cout<<"FedTree only supports histogram-based training yet";
        exit(1);
    }
    std::vector<float_type> scores;
    if(fl_param.mode == "hybrid"){
        LOG(INFO) << "start hybrid trainer";
        trainer.hybrid_fl_trainer(parties, server, fl_param);
        for(int i = 0; i < n_parties; i++){
            float_type score;
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset);
            else
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "ensemble"){
        trainer.ensemble_trainer(parties, server, fl_param);
        float_type score;
        if(use_global_test_set) {
            score = server.global_trees.predict_score(fl_param.gbdt_param, test_dataset);
            scores.push_back(score);
        }
        else
            for(int i = 0; i < n_parties; i++) {
                score = server.global_trees.predict_score(fl_param.gbdt_param, test_subsets[i]);
                scores.push_back(score);
            }
    }
    else if(fl_param.mode == "solo"){
        trainer.solo_trainer(parties, fl_param);
        float_type score;
        for(int i = 0; i < n_parties; i++){
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset);
            else
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "centralized"){

        if (fl_param.deltaboost_param.enable_delta) {
            auto deltaboost = std::unique_ptr<DeltaBoost>(new DeltaBoost());
            float_type score;
            deltaboost->train(fl_param.deltaboost_param, dataset);

            string model_path = string_format("cache/%s.model",
                                              fl_param.deltaboost_param.dataset_name.c_str());

            parser.save_model(model_path, fl_param.deltaboost_param, *deltaboost, dataset);
            parser.load_model(model_path, fl_param.deltaboost_param, *deltaboost, dataset);

            if(use_global_test_set) {
                score = deltaboost->predict_score(fl_param.deltaboost_param, test_dataset);
                scores.push_back(score);
            }
            else {
                for(int i = 0; i < n_parties; i++) {
                    score = deltaboost->predict_score(fl_param.deltaboost_param, test_subsets[i]);
                    scores.push_back(score);
                }
            }

            std::chrono::high_resolution_clock timer;
            auto start_rm = timer.now();
            int num_removals = static_cast<int>(fl_param.deltaboost_param.remove_ratio * dataset.n_instances());
            LOG(INFO) << num_removals << " samples to be removed from model";
            vector<int> removing_indices(static_cast<int>(fl_param.deltaboost_param.remove_ratio * dataset.n_instances()));
            std::iota(removing_indices.begin(), removing_indices.end(), 0);
            deltaboost->remove_samples(fl_param.deltaboost_param, dataset, removing_indices);
            auto stop_rm = timer.now();
            std::chrono::duration<float> removing_time = stop_rm - start_rm;
            LOG(INFO) << "removing time = " << removing_time.count();

            LOG(INFO) << "Predict after removals";
            if(use_global_test_set) {
                deltaboost->predict_score(fl_param.deltaboost_param, test_dataset);
            }
            else {
                for(int i = 0; i < n_parties; i++) {
                    deltaboost->predict_score(fl_param.deltaboost_param, test_subsets[i]);
                }
            }

        } else {
            auto gbdt = std::unique_ptr<GBDT>(new GBDT());
            gbdt->train(fl_param.gbdt_param, dataset);

            float_type score;
            if(use_global_test_set) {
                score = gbdt->predict_score(fl_param.gbdt_param, test_dataset);
                scores.push_back(score);
            }
            else {
                for(int i = 0; i < n_parties; i++) {
                    score = gbdt->predict_score(fl_param.gbdt_param, test_subsets[i]);
                    scores.push_back(score);
                }
            }
        }

    } else if (fl_param.mode == "vertical") {
        trainer.vertical_fl_trainer(parties, server, fl_param);
        float_type score;
//        if(use_global_test_set)
//        score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset);
        score = parties[0].gbdt.predict_score_vertical(fl_param.gbdt_param, test_dataset, batch_idxs);
//        else
//            score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_subsets[0]);
        scores.push_back(score);
    }else if (fl_param.mode == "horizontal") {
        LOG(INFO)<<"start horizontal training";
        trainer.horizontal_fl_trainer(parties, server, fl_param);
        LOG(INFO)<<"end horizontal training";
        float_type score;
        if(use_global_test_set)
            score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset);
        else
            score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_subsets[0]);
        scores.push_back(score);
    }
//        parser.save_model("global_model", fl_param.gbdt_param, server.global_trees.trees, dataset);
//    }
    return 0;
}
