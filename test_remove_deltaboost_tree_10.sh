n_rounds=$1  # 1, 10, 30, 100

for dataset in gisette; do
  ./test_remove_deltaboost.sh 10 $dataset 1e-03 $n_rounds
  ./test_remove_deltaboost.sh 10 $dataset 1e-02 $n_rounds
  ./test_remove_gbdt.sh 10 $dataset
done
wait
