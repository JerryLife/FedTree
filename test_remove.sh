mkdir -p out/remove_test

for ratio in 0.01 0.001; do
  for dataset in msd; do
    if [ $ratio = "0.01" ]; then
      taskset -c 24-96 ./main conf/"$dataset".conf data=./data/"$dataset".train remove_ratio="$ratio" \
        save_model_name="$dataset"_single_tree remain_data=./data/"$dataset".train.remain_1e-02 delete_data=./data/"$dataset".train.delete_1e-02 > \
        out/remove_test/"$dataset"_deltaboost_"$ratio".out &
      taskset -c 24-96 ./main conf/"$dataset".conf data=./data/"$dataset".train.remain_1e-02 remove_ratio="$ratio" \
          save_model_name="$dataset"_single_tree_remain_1e-02 remain_data=./data/"$dataset".train.remain_1e-02 delete_data=./data/"$dataset".train.delete_1e-02 > \
          out/remove_test/"$dataset"_deltaboost_"$ratio"_retrain.out &
    else
      if [ $ratio = "0.001" ]; then
        taskset -c 24-96 ./main conf/"$dataset".conf data=./data/"$dataset".train remove_ratio="$ratio" \
            save_model_name="$dataset"_single_tree remain_data=./data/"$dataset".train.remain_1e-03 delete_data=./data/"$dataset".train.delete_1e-03 > \
            out/remove_test/"$dataset"_deltaboost_"$ratio".out &
          taskset -c 24-96 ./main conf/"$dataset".conf data=./data/"$dataset".train.remain_1e-03 remove_ratio="$ratio" \
              save_model_name="$dataset"_single_tree_remain_1e-03 remain_data=./data/"$dataset".train.remain_1e-03 delete_data=./data/"$dataset".train.delete_1e-03 > \
              out/remove_test/"$dataset"_deltaboost_"$ratio"_retrain.out &
      fi
    fi
  done
  wait
done