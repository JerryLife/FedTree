import numpy as np
import pickle
import time

import xgboost as xgb

from train_test_split import load_data



class Record(object):

    def __init__(self, raw_data):
        self.raw_data = raw_data

    @classmethod
    def load_from_file(cls, dataset, version):
        """
        :param dataset:
        :param version: [1%, 0.1%]
        :return:
        """
        with open(f'../share/DART_data/{dataset}/{version}/data', 'rb') as f:
            record = cls(pickle.load(f))
            return record

    '''
    return a list of length 10, which is training seconds for seed from 0 to 9
    '''

    def train_times(self):
        return self.raw_data['training_seconds']

    '''
    return a list of length 10, which is forgetting seconds for seed from 0 to 9
    '''

    def forget_times(self):
        return self.raw_data['forgetting_seconds']

    '''
    return a list of length 10, which is retraining seconds for seed from 0 to 9
    '''

    def retrain_times(self):
        return self.raw_data['retraining_seconds']

    '''
    read real labels for input datasets
    dataset_type: from ['test', 'forget', 'retrain']
    '''

    def get_real_labels(self, dataset_type):
        return self.raw_data[f'{dataset_type}_data_df'][['real']]

    '''
    read data as dataframe to calculate the matrix
    model_type: from ['origin', 'forget', 'retrain']
    dataset_type: from ['test', 'forget', 'retrain']
    '''

    def read(self, model_type, dataset_type):
        return self.raw_data[f'{dataset_type}_data_df'].filter(regex=(f'{model_type}.*'))


def test_xgb(dataset, n_trees=10):
    dataset_path = f'../data/{dataset}.train'
    X, y = load_data(dataset_path, 'csv', scale_y=True, output_dense=True)
    dtrain = xgb.DMatrix(X, label=y)
    st = time.time()
    bst = xgb.train({'tree_method': 'approx', 'objective': 'binary:logistic', 'sketch_eps': 0.01}, dtrain, num_boost_round=n_trees)
    et = time.time()
    print(f'XGBoost training time: {et - st:.2f}s')
    return bst


if __name__ == '__main__':
    # record = Record.load_from_file('covtype', '1%')
    # print(record.raw_data['test_data_df'].describe())
    # print(record.raw_data['test_data_df'].columns)

    test_xgb('covtype', 10)

