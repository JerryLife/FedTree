from GBDT import GBDT, Tree, Node
from train_test_split import load_data
import ujson as json
import numpy as np

dataset = "codrna"
retrain = GBDT.load_from_json(json.load(open(f"../cache/{dataset}_tree10_retrain_1e-02_deltaboost.json")), 'deltaboost')
delete = GBDT.load_from_json(json.load(open(f"../cache/{dataset}_tree10_original_1e-02_deleted.json")), 'deltaboost')
for i in range(10):
    print(f"Tree{i}: {retrain.trees[i] == delete.trees[i]}")
