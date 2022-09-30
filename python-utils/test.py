from GBDT import GBDT, Tree, Node
from train_test_split import load_data
import ujson as json
import numpy as np

dataset = "codrna"
tree = 10
i = 5
remove = '1e-03'
retrain = GBDT.load_from_json(json.load(open(f"../cache/{dataset}_tree{tree}_retrain_{remove}_{i}_deltaboost.json")), 'deltaboost')
delete = GBDT.load_from_json(json.load(open(f"../cache/{dataset}_tree{tree}_original_{remove}_{i}_deleted.json")), 'deltaboost')
for i in range(tree):
    print(f"Tree{i}: {retrain.trees[i] == delete.trees[i]}")
