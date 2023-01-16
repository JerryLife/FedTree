n_trees=$1  # 1, 10, 30, 100
dataset=$2  # cadata, codrna, gisette, covtype, msd, higgs
ratio=$3    # 1e-02, 1e-03
i=$4        # 1, 2, 3, 4, 5
cpus="0-63"

subdir=tree"$n_trees"
outdir="out/mem/$subdir/"
mkdir -p $outdir

: "${1?"Number of trees unset."}"
: "${1?"Dataset unset."}"
: "${1?"Ratio unset."}"
: "${1?"Number of rounds unset."}"

taskset -c $cpus ./main conf/"$subdir"/"$dataset"_"$ratio".conf data=./data/"$dataset".train remove_ratio="$ratio" n_trees="$n_trees" \
    remain_data=./data/"$dataset".train.remain_"$ratio" delete_data=./data/"$dataset".train.delete_"$ratio" \
    save_model_name="$dataset"_tree"$n_trees"_original_"$ratio"_"$i" enable_delta=true seed="$i" perform_remove=false > \
    $outdir/"$dataset"_deltaboost_"$ratio"_"$i".out