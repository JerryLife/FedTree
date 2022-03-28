from GBDT import GBDT, Tree, Node
from train_test_split import load_data
import json
import numpy as np

dataset = "codrna"
gbdt = GBDT.load_from_json(json.load(open(f'../cache/{dataset}_gbdt.json')), 'gbdt')
deltaboost = GBDT.load_from_json(json.load(open(f'../cache/{dataset}.json')), 'deltaboost')

train_X, train_y = load_data(f"../data/{dataset}.train", data_fmt='libsvm', scale_y=False, output_dense=True)
gbdt_out = gbdt.predict_score(train_X)
deltaboost_out = deltaboost.predict_score(train_X)

# print(f"{(gbdt.trees[1] == deltaboost.trees[1])=}")
print(f"{(gbdt == deltaboost)=}")
