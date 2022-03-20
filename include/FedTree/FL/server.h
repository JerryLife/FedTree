//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_SERVER_H
#define FEDTREE_SERVER_H

#include "FedTree/FL/party.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
//#include "FedTree/Encryption/HE.h"
#include "FedTree/DP/noises.h"
#include "FedTree/Tree/gbdt.h"
#include "omp.h"

// Todo: the server structure.

class Server : public Party {
public:
    void init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party);

    void horizontal_init (FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, DataSet &dataSet);

    void vertical_init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, vector<float_type> y,
                       vector<float_type> label);

    void propose_split_candidates();
    void send_info(string info_type);
//    void send_info(vector<Party> &parties, AdditivelyHE::PaillierPublicKey serverKey,vector<SplitCandidate>candidates);
    void sum_histograms();
    void hybrid_merge_trees();
    void ensemble_merge_trees();

    void sample_data();
    void predict_raw_vertical_jointly_in_training(const GBDTParam &model_param, vector<Party> &parties,
                                                  SyncArray<float_type> &y_predict);
    GBDT global_trees;
    vector<GBDT> local_trees;
    GBDTParam model_param;
    vector<int> n_instances_per_party;
    

//    AdditivelyHE::PaillierPublicKey publicKey;
//    vector<AdditivelyHE::PaillierPublicKey> pk_vector;
    Paillier paillier;

    void send_key(Party &party) {
        party.paillier = paillier;
    }

    void homo_init() {
        paillier = Paillier(512);
    }

    void decrypt_gh(GHPair &gh) {
        gh.homo_decrypt(paillier);
    }

    void decrypt_gh_pairs(SyncArray<GHPair> &encrypted) {

#ifdef USE_CUDA
        device_loop(encrypted.size(), [=] __device__(int idx){

        });
#else
        auto encrypted_data = encrypted.host_data();
        #pragma omp parallel for
        for (int i = 0; i < encrypted.size(); i++) {
            encrypted_data[i].homo_decrypt(paillier);
        }
#endif
    }

    void encrypt_gh_pairs(SyncArray<GHPair> &raw) {
#ifdef USE_CUDA
        pl_gpu.encrypt(raw);
        auto raw_data = raw.device_data();
        cgbn_error_report_t *report;
        CUDA_CHECK(cgbn_error_report_alloc(&report));
        device_loop(raw.size(), [=] __device__(int idx){
            context_t bn_context(cgbn_report_monitor, report, idx);
            env_t bn_env(bn_context.env<env_t>());
            env_t::cgbn_t
        })
#else
        auto raw_data = raw.host_data();
        #pragma omp parallel for
        for (int i = 0; i < raw.size(); i++) {
            raw_data[i].homo_encrypt(paillier);
        }
#endif
    }

private:
//    std::unique_ptr<TreeBuilder> fbuilder;
    DPnoises<double> DP;
};

#endif //FEDTREE_SERVER_H
