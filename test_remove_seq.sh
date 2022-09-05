n_trees=$1
cpus="0-90"

: ${1?"Number of trees unset."}

subdir=tree"$n_trees"
outdir="out/remove_test/$subdir/"
mkdir -p $outdir

for dataset in codrna cadata covtype gisette msd; do
  for ratio in 0.01 0.001; do
    if [ $ratio = "0.01" ]; then
      taskset -c $cpus ./main conf/"$dataset"_1e-02.conf data=./data/"$dataset".train remove_ratio="$ratio" n_trees=$n_trees \
        save_model_name="$dataset"_tree"$n_trees"_original_1e-02 remain_data=./data/"$dataset".train.remain_1e-02 delete_data=./data/"$dataset".train.delete_1e-02 > \
        $outdir/"$dataset"_deltaboost_"$ratio".out
      taskset -c $cpus ./main conf/"$dataset"_1e-02.conf data=./data/"$dataset".train.remain_1e-02 remove_ratio="$ratio" n_trees=$n_trees \
          save_model_name="$dataset"_tree"$n_trees"_retrain_1e-02 remain_data=./data/"$dataset".train.remain_1e-02 delete_data=./data/"$dataset".train.delete_1e-02 > \
          $outdir/"$dataset"_deltaboost_"$ratio"_retrain.out
#      taskset -c $cpus ./main conf/"$dataset"_1e-02.conf save_model_name="$dataset"_tree"$n_trees"_gbdt data=./data/"$dataset".train enable_delta=false n_trees=$n_trees > \
#        $outdir/"$dataset"_gbdt.out
    else
      if [ $ratio = "0.001" ]; then
        taskset -c $cpus ./main conf/"$dataset"_1e-03.conf data=./data/"$dataset".train remove_ratio="$ratio" n_trees=$n_trees \
            save_model_name="$dataset"_tree"$n_trees"_original_1e-03 remain_data=./data/"$dataset".train.remain_1e-03 delete_data=./data/"$dataset".train.delete_1e-03 > \
            $outdir/"$dataset"_deltaboost_"$ratio".out
          taskset -c $cpus ./main conf/"$dataset"_1e-03.conf data=./data/"$dataset".train.remain_1e-03 remove_ratio="$ratio" n_trees=$n_trees \
              save_model_name="$dataset"_tree"$n_trees"_retrain_1e-03 remain_data=./data/"$dataset".train.remain_1e-03 delete_data=./data/"$dataset".train.delete_1e-03 > \
              $outdir/"$dataset"_deltaboost_"$ratio"_retrain.out
      fi
    fi
  done
  wait
done