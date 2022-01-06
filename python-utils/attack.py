from typing import Callable
import json
import abc

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData, SlicingSpec, AttackType

import numpy as np
from sklearn.metrics import log_loss

from GBDT import GBDT, Evaluative
from train_test_split import load_data


def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    return - true_label * np.log(p) - (1 - true_label) * np.log(1 - p)


class ModelEvaluator:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate(self, model: Evaluative):
        score_train = model.predict_score(self.x_train)
        score_test = model.predict_score(self.x_test)
        loss_train = logloss(self.y_train, score_train)
        assert np.mean(loss_train) == log_loss(self.y_train, score_train)
        loss_test = logloss(self.y_test, score_test)
        assert np.mean(loss_test) == log_loss(self.y_test, score_test)

        input_data = AttackInputData(
            loss_train=loss_train,
            loss_test=loss_test,
            labels_train=self.y_train,
            labels_test=self.y_test
        )
        slicing_spec = SlicingSpec(
            entire_dataset=True,
            by_class=True,
            by_percentiles=False,
            by_classification_correctness=True
        )
        attack_types = [
            AttackType.THRESHOLD_ATTACK,
            AttackType.LOGISTIC_REGRESSION
        ]
        attack_result = mia.run_attacks(attack_input=input_data, slicing_spec=slicing_spec, attack_types=attack_types)
        pass


class ModelLoader:
    def __init__(self):
        pass

    def load_from_json(self, model_path, type='deltaboost') -> Callable:
        assert type in ['deltaboost', 'gbdt']
        with open(model_path, 'r') as f:
            js = json.load(f)
        gbdt = GBDT.load_from_json(js, type)
        print(f"Loaded from {model_path}")

        return gbdt


if __name__ == '__main__':
    loader = ModelLoader()
    model = loader.load_from_json("../cache/cod-rna.json")
    remain_X, remain_y = load_data("../data/codrna.train.remain_1e-02", data_fmt='libsvm')
    delete_X, delete_y = load_data("../data/codrna.train.delete_1e-02", data_fmt='libsvm')
    evaluator = ModelEvaluator(remain_X, delete_X, remain_y, delete_y)
    evaluator.evaluate(model)

