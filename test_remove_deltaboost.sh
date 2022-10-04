n_trees=$1  # 1, 10, 30, 100
dataset=$2  # cadata, codrna, gisette, covtype, msd, higgs
ratio=$3    # 1e-02, 1e-03
n_rounds=$4 # 1, 5, 10
cpus="0-71"

: "${1?"Number of trees unset."}"
: "${1?"Dataset unset."}"
: "${1?"Ratio unset."}"
: "${1?"Number of rounds unset."}"

subdir=tree"$n_trees"
outdir="out/remove_test/$subdir/"
mkdir -p $outdir


# iterate n_rounds times
for i in $(seq 0 $n_rounds); do
  taskset -c $cpus ./main conf/"$subdir"/"$dataset"_"$ratio".conf data=./data/"$dataset".train remove_ratio="$ratio" n_trees="$n_trees" \
    remain_data=./data/"$dataset".train.remain_"$ratio" delete_data=./data/"$dataset".train.delete_"$ratio" \
    save_model_name="$dataset"_tree"$n_trees"_original_"$ratio"_"$i" seed="$i" > \
    $outdir/"$dataset"_deltaboost_"$ratio"_"$i".out
done
wait

for i in $(seq 0 $n_rounds); do
  taskset -c $cpus ./main conf/"$subdir"/"$dataset"_"$ratio".conf data=./data/"$dataset".train.remain_"$ratio" remove_ratio="$ratio" n_trees="$n_trees" perform_remove=false \
    remain_data=./data/"$dataset".train.remain_"$ratio" delete_data=./data/"$dataset".train.delete_"$ratio" \
    save_model_name="$dataset"_tree"$n_trees"_retrain_"$ratio"_"$i" seed="$i"   > \
    $outdir/"$dataset"_deltaboost_"$ratio"_retrain_"$i".out
done
wait



