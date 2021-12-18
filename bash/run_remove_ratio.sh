#!/usr/bin/env bash


for dataset in covtype
do
  for ratio in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5
  do
    taskset -c 1-90 nohup ./main conf/"$dataset".conf remove_ratio=$ratio > out/remove_ratio/"$dataset"_remove_"$ratio".out &
  done
done