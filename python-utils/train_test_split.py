from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import argparse


def load_data(data_path, data_fmt, scale_y=False) -> tuple:
    """
    :param scale_y:
    :param data_fmt: data format (e.g. libsvm)
    :param data_path: path of the data
    :return: data, labels
    """
    data_fmt = data_fmt.lower()
    assert data_fmt in ['libsvm'], "Unsupported format"

    if data_fmt == 'libsvm':
        X, y = load_svmlight_file(data_path)
        print("Got X with shape {}, y with shape {}".format(X.shape, y.shape))
    else:
        assert False

    if scale_y:
        y = MinMaxScaler((0, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1)

    return X, y


def split_data(data, labels, val_rate=0.1, test_rate=0.2, seed=0):
    print("Splitting...")
    indices = np.arange(data.shape[0])
    if np.isclose(val_rate, 0.0):
        train_data, test_data, train_labels, test_labels, train_idx, test_idx = \
            train_test_split(data, labels, indices, test_size=test_rate, shuffle=True, random_state=seed)
        return train_data, None, test_data, train_labels, None, test_labels, train_idx, None, test_idx
    elif np.isclose(test_rate, 0.0):
        train_data, val_data, train_labels, val_labels, train_idx, val_idx = \
            train_test_split(data, labels, indices, test_size=val_rate, shuffle=True, random_state=seed)
        return train_data, val_data, None, train_labels, val_labels, None, train_idx, val_idx, None
    else:
        train_val_data, test_data, train_val_labels, test_labels, train_val_idx, test_idx = \
            train_test_split(data, labels, indices, test_size=test_rate, shuffle=True, random_state=seed)
        split_val_rate = val_rate / (1. - test_rate)
        train_data, val_data, train_labels, val_labels, train_idx, val_idx = \
            train_test_split(train_val_data, train_val_labels, train_val_idx, shuffle=True, test_size=split_val_rate,
                             random_state=seed)
        return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_idx, val_idx, test_idx


def save_data(X, y, save_path, save_fmt='libsvm'):
    data_fmt = save_fmt.lower()
    assert data_fmt in ['libsvm'], "Unsupported format"

    if data_fmt == 'libsvm':
        dump_svmlight_file(X, y, save_path, zero_based=False)
    else:
        assert False

    print("Saved {} data to {}".format(X.shape, save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('-if', '--input-fmt', type=str, default='libsvm')
    parser.add_argument('-of', '--output-fmt', type=str, default='libsvm')
    parser.add_argument('--scale-y', action='store_true')
    parser.add_argument('-v', '--val-rate', type=float, default=0.1)
    parser.add_argument('-t', '--test-rate', type=float, default=0.2)
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    X, y = load_data(data_path=args.data_path, data_fmt=args.input_fmt, scale_y=args.scale_y)
    train_X, val_X, test_X, train_y, val_y, test_y, _, _, _ = split_data(
        X, y, val_rate=args.val_rate, test_rate=args.test_rate, seed=args.seed)

    save_data(train_X, train_y, save_path=args.data_path + ".train", save_fmt=args.output_fmt)
    if not np.isclose(args.val_rate, 0):
        save_data(val_X, val_y, save_path=args.data_path + ".val", save_fmt=args.output_fmt)
    if not np.isclose(args.test_rate, 0):
        save_data(test_X, test_y, save_path=args.data_path + ".test", save_fmt=args.output_fmt)
