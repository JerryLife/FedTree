#!/usr/bin/env bash

n_rounds=$1  # 1, 10, 30, 100

for dataset in covtype; do
  ./test_remove_deltaboost.sh 30 $dataset 1e-03 $n_rounds
  ./test_remove_deltaboost.sh 30 $dataset 1e-02 $n_rounds
  ./test_remove_gbdt.sh 30 $dataset
done
wait
