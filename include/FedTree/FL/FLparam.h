//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_FLPARAM_H
#define FEDTREE_FLPARAM_H

#include "FedTree/Tree/GBDTparam.h"
#include "FedTree/common.h"

// Todo: automatically set partition
class FLParam {
public:
    int n_parties; // number of parties
    bool partition; // input a single dataset for partitioning or input datasets for each party.
    float alpha; //the concentration parameter of Dir based partition approaches.
    int n_hori; //the number of horizontal partitioning subsets in hybrid partition.
    int n_verti; //the number of vertical partitioning subsets in hybrid partition.
    string mode; // "horizontal", "vertical", "hybrid", or "centralized"
    string partition_mode; // "horizontal", "vertical" or "hybrid"
    string privacy_tech; //"none" or "he" (homomorphic encryption) or "dp" (differential privacy) or "sa" (secure aggregation)
    string propose_split; // "server" or "client"
    string merge_histogram; // "server" or "client"
    float variance; // variance of dp noise if privacy_tech=="dp"
    float privacy_budget;       // privacy budget for differential privacy
    string ip_address; // IP address of the server
    float ins_bagging_fraction; // randomly sample subset to train a tree without replacement
    int seed; // random seed for partitioning
    string data_format; // data format: "libsvm" or "csv"
    string label_location; // "server" or "party" for vertical FL
    int n_features; //specify the number of features for horizontal FL with sparse datasets
    GBDTParam gbdt_param; // parameters for the gbdt training
};


#endif //FEDTREE_FLPARAM_H
