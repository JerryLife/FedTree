from GBDT import GBDT, Tree, Node
from train_test_split import load_data
import ujson as json
import numpy as np

dataset = "codrna"
tree = 30
remove = '1e-03'

cnt_rvd_equal = 0
n_rounds = 10
for i in range(1, n_rounds):
    retrain = GBDT.load_from_json(json.load(open(
        f"../_cache/{dataset}_tree{tree}_retrain_{remove}_{i}_deltaboost.json")), 'deltaboost')
    delete = GBDT.load_from_json(json.load(open(f"../_cache/{dataset}_tree{tree}_original_{remove}_{i}_deleted.json")), 'deltaboost')
    originl = GBDT.load_from_json(json.load(open(
        f"../_cache/{dataset}_tree{tree}_original_{remove}_{i}_deltaboost.json")), 'deltaboost')
    # retrain = GBDT.load_from_json(json.load(open(f"../cache/{dataset}_tree{tree}_retrain_{remove}_deltaboost.json")), 'deltaboost')
    # delete = GBDT.load_from_json(json.load(open(f"../cache/{dataset}_tree{tree}_original_{remove}_deleted.json")), 'deltaboost')

    for j in range(tree):
        is_rvd_equal = retrain.trees[j] == delete.trees[j]
        is_rvo_equal = retrain.trees[j] == originl.trees[j]
        print(f"Tree{i}: {is_rvd_equal}, {is_rvo_equal}")
        if is_rvd_equal:
            cnt_rvd_equal += 1
print(f"Equal: {cnt_rvd_equal}/{n_rounds}")

