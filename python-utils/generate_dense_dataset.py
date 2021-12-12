from sklearn.datasets import load_svmlight_file, dump_svmlight_file, make_classification
import numpy as np
import argparse

from train_test_split import save_data


def generate_dense_dataset(save_path):
    X, y = make_classification(n_samples=10000, n_features=20, shift=1)
    assert not np.isclose(X, 0).any()
    save_data(X, y, save_path, 'libsvm')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    args = parser.parse_args()
    generate_dense_dataset(save_path=args.data_path)

