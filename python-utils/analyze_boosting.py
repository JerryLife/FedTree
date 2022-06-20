import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import ujson as json

from GBDT import GBDT
from train_test_split import load_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_gradient(y, y_p):
    p = sigmoid(y_p)
    gradient = p - y
    hessian = p * (1 - p)
    return gradient, hessian


def plot_gradient_diff(dataset, n_trees, remove_ratio: str, output_dir):
    model_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deltaboost.json"
    model_remain_path = f"../cache/{dataset}_tree{n_trees}_retrain_{remove_ratio}_deltaboost.json"
    model_deleted_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deleted.json"
    with open(model_path, 'r') as f1, open(model_remain_path, 'r') as f2, open(model_deleted_path) as f3:
        js1 = json.load(f1)
        js2 = json.load(f2)
        js3 = json.load(f3)
    gbdt = GBDT.load_from_json(js1, 'deltaboost')
    gbdt_remain = GBDT.load_from_json(js2, 'deltaboost')
    gbdt_deleted = GBDT.load_from_json(js3, 'deltaboost')
    # train_X, train_y = load_data(f"../data/{dataset}.train", data_fmt='libsvm', output_dense=True)
    remain_X, remain_y = load_data(f"../data/{dataset}.train.remain_{remove_ratio}", data_fmt='libsvm',
                                   output_dense=True)

    # plt.style.use('seaborn-deep')
    for i in range(1, n_trees):
        score = gbdt.predict_score(remain_X, n_used_trees=i)
        g, h = get_gradient(remain_y, score)
        score_remain = gbdt_remain.predict_score(remain_X, n_used_trees=i)
        g_remain, h_remain = get_gradient(remain_y, score_remain)
        score_deleted = gbdt_deleted.predict_score(remain_X, n_used_trees=i)
        g_deleted, h_deleted = get_gradient(remain_y, score_deleted)

        sorted_g = g[g.argsort()]
        delta_g_retrain = np.abs(g_deleted - g_remain)
        sorted_delta_g_retrain = delta_g_retrain[g.argsort()]
        delta_g_original = np.abs(g - g_deleted)
        sorted_delta_g_original = delta_g_original[g.argsort()]

        # plot both figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=None, sharey=True)
        ax1.scatter(sorted_g, sorted_delta_g_original, s=3)
        ax1.set_xlabel('Values of gradients')
        ax1.set_ylabel('Absolute difference')
        ax1.set_title('(1) deleted vs. original')
        ax2.scatter(sorted_g, sorted_delta_g_retrain, s=3)
        ax2.set_xlabel('Values of gradients')
        ax2.set_ylabel('Absolute difference')
        ax2.set_title('(b) deleted vs. retrain')
        fig.suptitle(f'{dataset}, remove {remove_ratio}: absolute difference of gradients after tree{i}')
        dataset_dir = os.path.join(output_dir, f"{dataset}")
        os.makedirs(dataset_dir, exist_ok=True)
        output_path = os.path.join(dataset_dir, f"{dataset}_{remove_ratio}_tree{i:02d}.jpg")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved to {output_path}.")


def plot_score_diff(dataset, n_trees, remove_ratio: str, output_dir, print_metric=None):
    model_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deltaboost.json"
    model_remain_path = f"../cache/{dataset}_tree{n_trees}_retrain_{remove_ratio}_deltaboost.json"
    model_deleted_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deleted.json"
    with open(model_path, 'r') as f1, open(model_remain_path, 'r') as f2, open(model_deleted_path) as f3:
        js1 = json.load(f1)
        js2 = json.load(f2)
        js3 = json.load(f3)
    gbdt = GBDT.load_from_json(js1, 'deltaboost')
    gbdt_remain = GBDT.load_from_json(js2, 'deltaboost')
    gbdt_deleted = GBDT.load_from_json(js3, 'deltaboost')
    # train_X, train_y = load_data(f"../data/{dataset}.train", data_fmt='libsvm', output_dense=True)
    remain_X, remain_y = load_data(f"../data/{dataset}.train.remain_{remove_ratio}", data_fmt='libsvm',
                                   output_dense=True)

    # plt.style.use('seaborn-deep')
    for i in range(1, n_trees + 1):
        score = gbdt.predict_score(remain_X, n_used_trees=i)
        # g, h = get_gradient(remain_y, score)
        score_remain = gbdt_remain.predict_score(remain_X, n_used_trees=i)
        # g_remain, h_remain = get_gradient(remain_y, score_remain)
        score_deleted = gbdt_deleted.predict_score(remain_X, n_used_trees=i)
        # g_deleted, h_deleted = get_gradient(remain_y, score_deleted)

        # sorted_score = score[score.argsort()]
        delta_score_original = np.abs(score_deleted - score)
        # sorted_delta_score_original = delta_score_original[score.argsort()]
        delta_score_deleted = np.abs(score_deleted - score_remain)
        # sorted_delta_score_deleted = delta_score_deleted[score.argsort()]

        # plot both figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=None, sharey=True)
        ax1.scatter(score, delta_score_original, s=3)
        ax1.set_xlabel('Values of predictions')
        ax1.set_ylabel('Absolute difference')
        ax1.set_title('(a) delete vs. original')
        ax2.scatter(score, delta_score_deleted, s=3)
        ax2.set_xlabel('Values of gradients')
        ax2.set_ylabel('Absolute difference')
        ax2.set_title('(b) deleted vs. retrain')
        fig.suptitle(f'{dataset}, remove {remove_ratio}: absolute difference of prediction after tree{i}')
        dataset_dir = os.path.join(output_dir, f"{dataset}")
        os.makedirs(dataset_dir, exist_ok=True)
        output_path = os.path.join(dataset_dir, f"{dataset}_{remove_ratio}_tree{i:02d}.jpg")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved to {output_path}.")

        # print error
        if print_metric == 'error':
            error_remain = np.count_nonzero(((sigmoid(score_remain) > 0.5) + 0.) != remain_y) / remain_y.shape[0]
            error_deleted = np.count_nonzero(((sigmoid(score_deleted) > 0.5) + 0.) != remain_y) / remain_y.shape[0]
            print(f"{error_remain=}, {error_deleted=}")
        elif print_metric == 'rmse':
            rmse_remain = np.sqrt(mean_squared_error(remain_y, score_remain))
            rmse_deleted = np.sqrt(mean_squared_error(remain_y, score_deleted))
            print(f"{rmse_remain=}, {rmse_deleted=}")
        else:
            assert print_metric is None


if __name__ == '__main__':
    n_trees = 10
    # for dataset, metric in zip(['codrna', 'cadata', 'covtype', 'gisette', 'msd'],
    #                            ['error', 'rmse', 'error', 'error', 'rmse']):
    #     for remove_ratio in ['1e-03', '1e-02']:
    #         output_dir = "fig/prediction_diff"
    #         plot_gradient_diff(dataset, n_trees, remove_ratio, output_dir)

    dataset = 'covtype'
    remove_ratio = '1e-03'
    # output_g_dir = "fig/gradient_diff"
    # plot_gradient_diff(dataset, n_trees, remove_ratio, output_g_dir)
    output_p_dir = "../../DeltaBoost-Python/fig/prediction_diff"
    plot_score_diff(dataset, n_trees, remove_ratio, output_p_dir)