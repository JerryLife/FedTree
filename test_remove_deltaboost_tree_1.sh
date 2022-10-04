#!/usr/bin/env bash

n_trees=$1  # 1, 10, 30, 100

for dataset in cadata codrna gisette covtype msd; do
  ./test_remove_deltaboost.sh 1 $dataset 1e-03 $n_trees ;
  ./test_remove_deltaboost.sh 1 $dataset 1e-02 $n_trees ;
done
