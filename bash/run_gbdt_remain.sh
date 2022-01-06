#!/usr/bin/env bash


for dataset in cadata codrna covtype gisette
do
  for pair in 2,0.01 3,0.001
  do
    IFS=',' read id ratio <<< "${pair}"
    taskset -c 1-90 nohup ./main conf/"$dataset".conf enable_delta=false data=./data/"$dataset".train.remain_1e-0"$id" \
    remain_data=./data/"$dataset".train.remain_1e-0"$id" delete_data=./data/"$dataset".train.delete_1e-0"$id" \
     > out/remove_ratio/"$dataset"_gbdt_"$ratio".out &
  done
done
