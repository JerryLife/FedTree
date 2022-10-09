n_trees=$1
dataset=$2
n_rounds=$3
cpus="0-63"

: ${1?"Number of trees unset."}
: ${1?"Dataset unset."}
: ${1?"Number of rounds unset."}

subdir=tree"$n_trees"
outdir="out/remove_test/$subdir/"
mkdir -p $outdir

for i in $(seq 0 $n_rounds); do
  for ratio in 0.001 0.01; do
    if [ $ratio = "0.01" ]; then
      taskset -c $cpus ./main conf/tree"$n_trees"/"$dataset"_1e-02.conf data=./data/"$dataset".train remove_ratio="$ratio" n_trees=$n_trees \
        save_model_name="$dataset"_tree"$n_trees"_original_1e-02 remain_data=./data/"$dataset".train.remain_1e-02 delete_data=./data/"$dataset".train.delete_1e-02 \
        seed="$i" \
        > $outdir/"$dataset"_deltaboost_"$ratio".out
      taskset -c $cpus ./main conf/tree"$n_trees"/"$dataset"_1e-02.conf data=./data/"$dataset".train.remain_1e-02 remove_ratio="$ratio" n_trees=$n_trees performe_remove=false \
          save_model_name="$dataset"_tree"$n_trees"_retrain_1e-02 remain_data=./data/"$dataset".train.remain_1e-02 delete_data=./data/"$dataset".train.delete_1e-02 \
          seed="$i" \
          > $outdir/"$dataset"_deltaboost_"$ratio"_retrain.out
    else
      if [ $ratio = "0.001" ]; then
        taskset -c $cpus ./main conf/tree"$n_trees"/"$dataset"_1e-03.conf data=./data/"$dataset".train remove_ratio="$ratio" n_trees=$n_trees \
            save_model_name="$dataset"_tree"$n_trees"_original_1e-03 remain_data=./data/"$dataset".train.remain_1e-03 delete_data=./data/"$dataset".train.delete_1e-03 \
            seed="$i" \
            > $outdir/"$dataset"_deltaboost_"$ratio".out
        taskset -c $cpus ./main conf/tree"$n_trees"/"$dataset"_1e-03.conf data=./data/"$dataset".train.remain_1e-03 remove_ratio="$ratio" n_trees=$n_trees performe_remove=false \
            save_model_name="$dataset"_tree"$n_trees"_retrain_1e-03 remain_data=./data/"$dataset".train.remain_1e-03 delete_data=./data/"$dataset".train.delete_1e-03 \
            seed="$i" \
            > $outdir/"$dataset"_deltaboost_"$ratio"_retrain.out
        taskset -c $cpus ./main conf/tree"$n_trees"/"$dataset"_1e-02.conf save_model_name="$dataset"_tree"$n_trees"_gbdt data=./data/"$dataset".train enable_delta=false n_trees=$n_trees \
              seed="$i" \
              > $outdir/"$dataset"_gbdt.out
      fi
    fi
  done
done
