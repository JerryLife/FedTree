for alpha in 0.0 0.2 0.4 0.6 0.8 1.0; do
  for dataset in codrna cadata covtype gisette msd; do
    taskset -c 25-90 ./main conf/"$dataset".conf save_model_name=alpha_"$alpha"/"$dataset"_single_tree data=./data/"$dataset".train alpha="$alpha" &
    taskset -c 25-90 ./main conf/"$dataset".conf save_model_name=alpha_"$alpha"/"$dataset"_single_tree_remain data=./data/"$dataset".train.remain_1e-02 alpha="$alpha" &
  done
  wait
done