for dataset in codrna cadata covtype gisette msd; do
  taskset -c 25-90 ./main conf/"$dataset".conf save_model_name="$dataset"_single_tree data=./data/"$dataset".train &
  taskset -c 25-90 ./main conf/"$dataset".conf save_model_name="$dataset"_single_tree_remain data=./data/"$dataset".train.remain_1e-02 &
done