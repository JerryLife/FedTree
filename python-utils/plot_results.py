import os.path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ujson as json

from GBDT import GBDT
from train_test_split import load_data


def get_scores_from_file(file_path, out_fmt='str') -> tuple[str, list]:
    with open(file_path, 'r') as f:
        f_text = f.read()
        metric = re.match(".*INFO (?:deltaboost|gbdt)\.cpp:[0-9]* : Test: (.*) =", f_text)
        pattern = re.compile(
            ".*INFO (?:deltaboost|gbdt)\.cpp:[0-9]* :.*(?:error|RMSE) = ([+-]?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)")
        scores = pattern.findall(f_text)

        if out_fmt == 'float':
            scores = [float(s) for s in scores]
        return metric, scores


def plot_score_before_after_removal(out_dir, datasets, remove_ratios, save_path=None, plot_type='table'):
    assert plot_type in ['raw', 'fig', 'table']
    summary = []
    for dataset in datasets:
        df_ratios = []
        for raw_ratio in remove_ratios:
            ratio = "0.01" if raw_ratio == '1e-02' else "0.001"
            out_path = os.path.join(out_dir, f"{dataset}_deltaboost_{ratio}.out")
            out_retrain_path = os.path.join(out_dir, f"{dataset}_deltaboost_{ratio}_retrain.out")
            metric, out_data = get_scores_from_file(out_path, out_fmt='float')
            assert len(out_data) == 6, f"{out_path}, got {len(out_data)}"
            scores = [float(x) for x in out_data[:3]]
            scores_del = [float(x) for x in out_data[3:]]
            _, out_retrain_data = get_scores_from_file(out_retrain_path, out_fmt='float')
            scores_retrain = out_retrain_data[:3]

            # reorder the result
            scores = np.array([scores[2], scores[1], scores[0]])
            scores_del = np.array([scores_del[2], scores_del[1], scores_del[0]])
            scores_retrain = np.array([scores_retrain[2], scores_retrain[1], scores_retrain[0]])

            x_labels = [r'$D_r$', r'$D_f$', r'$D_{test}$']
            df = pd.DataFrame(data={'Original': scores, 'Remove (ours)': scores_del,
                                    'Retrain (target)': scores_retrain})
            if plot_type == 'fig':
                df.set_index(x_labels)
                ax = df.plot(kind='bar', rot=0, xlabel="Datset partition", ylabel=metric,
                             title=f"{dataset} removing {ratio}")
                ax.legend()
                ax.margins(y=0.3)
                plt.show()
            elif plot_type in ['table']:
                columns = df.columns
                df = df.T
                df['Dataset'] = [f"\\multirow{{3}}{{*}}{{{dataset}}}", "", ""]
                df.rename(columns=dict(zip(range(len(x_labels)), x_labels)), inplace=True)
                df.set_index('Dataset', inplace=True)
                df_ratios.append(df)
        df_combine = pd.concat(df_ratios, axis=1)
        df_combine.insert(0, 'Method', columns)

        df_combine.style.format(lambda x: f"{x:.6f}")
        # df_combine = df_combine[['Method', r'$D$', r'$D_f$', r'$D_r$', r'$D$', r'$D_f$', r'$D_r$']]
        summary.append('\n'.join(df_combine.style.to_latex().split('\n')[3:6]))
    if plot_type == 'table':
        latex_table = '\n\\midrule\n'.join(summary)
        print(latex_table)


def plot_deltaboost_vs_gbdt(out_dir, datasets, save_path=None, present='test', n_trees=50):
    assert present in ['test', 'all']
    summary = []
    for dataset in datasets:
        ratio = 0.01  # either should be the same
        out_deltaboost_path = os.path.join(out_dir, f"tree{n_trees}/{dataset}_deltaboost_{ratio}.out")
        out_gbdt_path = os.path.join(out_dir, f"tree{n_trees}/{dataset}_gbdt.out")
        metric, deltaboost_data = get_scores_from_file(out_deltaboost_path, out_fmt='float')
        _, gbdt_data = get_scores_from_file(out_gbdt_path, out_fmt='float')
        deltaboost_scores = deltaboost_data[:3]
        gbdt_scores = gbdt_data

        # empty datasets
        if len(deltaboost_scores) == 0 or len(gbdt_scores) == 0:
            if present == 'test':
                summary.append([0, 0])
            else:
                assert False
        else:
            if present == 'test':
                summary.append([gbdt_scores[0], deltaboost_scores[0]])
            else:
                assert False
    # plot
    if present == 'test':
        summary_df = pd.DataFrame(data=summary, index=datasets, columns=['GBDT', 'DeltaBoost'])
        ax = summary_df.plot(kind='bar', rot=0, xlabel='Datasets', ylabel='Error/RMSE',
                             title=f'Error/RMSE of DeltaBoost and GBDT (tree {n_trees})',
                             figsize=None)
        ax.margins(y=0.1)
        ax.legend()
        plt.show()


def summary_model_diff_for_dataset(dataset, n_trees, remove_ratios):
    """
    Evaluate the model difference by inference. This function can be very slow.
    :param dataset:
    :param n_trees:
    :param remove_ratios:
    :return:
    """
    df_ratios = []
    for remove_ratio in remove_ratios:
        print(f"Remove ratio {remove_ratio}:")
        print("Loading models.")
        model_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deltaboost.json"
        model_remain_path = f"../cache/{dataset}_tree{n_trees}_retrain_{remove_ratio}_deltaboost.json"
        model_deleted_path = f"../cache/{dataset}_tree{n_trees}_original_{remove_ratio}_deleted.json"
        with open(model_path, 'r') as f1, open(model_remain_path, 'r') as f2, open(model_deleted_path) as f3:
            js1 = json.load(f1)
            js2 = json.load(f2)
            js3 = json.load(f3)
        gbdt_original = GBDT.load_from_json(js1, 'deltaboost')
        gbdt_retrain = GBDT.load_from_json(js2, 'deltaboost')
        gbdt_deleted = GBDT.load_from_json(js3, 'deltaboost')

        print("Loading dataset.")
        # train_X, train_y = load_data(f"../data/{dataset}.train", data_fmt='libsvm', output_dense=True)
        remain_X, remain_y = load_data(f"../data/{dataset}.train.remain_{remove_ratio}", data_fmt='csv',
                                       output_dense=True)
        delete_X, delete_y = load_data(f"../data/{dataset}.train.delete_{remove_ratio}", data_fmt='csv',
                                       output_dense=True)
        test_X, test_y = load_data(f"../data/{dataset}.test", data_fmt='csv', output_dense=True)

        print("Prediction on remain")
        original_score_on_remain = gbdt_original.predict_score(remain_X)
        retrain_score_on_remain = gbdt_retrain.predict_score(remain_X)
        deleted_score_on_remain = gbdt_deleted.predict_score(remain_X)
        diff_retrain_original_on_remain = np.average(np.abs(original_score_on_remain - retrain_score_on_remain))
        diff_retrain_delete_on_remain = np.average(np.abs(retrain_score_on_remain - deleted_score_on_remain))
        print("Prediction on delete")
        original_score_on_delete = gbdt_original.predict_score(delete_X)
        retrain_score_on_delete = gbdt_retrain.predict_score(delete_X)
        deleted_score_on_delete = gbdt_deleted.predict_score(delete_X)
        diff_retrain_original_on_delete = np.average(np.abs(original_score_on_delete - retrain_score_on_delete))
        diff_retrain_delete_on_delete = np.average(np.abs(retrain_score_on_delete - deleted_score_on_delete))
        print("Prediction on test")
        original_score_on_test = gbdt_original.predict_score(test_X)
        retrain_score_on_test = gbdt_retrain.predict_score(test_X)
        deleted_score_on_test = gbdt_deleted.predict_score(test_X)
        diff_retrain_original_on_test = np.average(np.abs(original_score_on_test - retrain_score_on_test))
        diff_retrain_delete_on_test = np.average(np.abs(retrain_score_on_test - deleted_score_on_test))

        x_labels = [r'$D_f$', r'$D_r$', r'$D_{test}$']
        df = pd.DataFrame(data={
            'original vs. retrain': [diff_retrain_original_on_remain, diff_retrain_original_on_delete,
                                     diff_retrain_original_on_test],
            'delete vs. retrain': [diff_retrain_delete_on_remain, diff_retrain_delete_on_delete,
                                   diff_retrain_delete_on_test]})
        columns = df.columns
        df = df.T
        df['Dataset'] = [f"\\multirow{{2}}{{*}}{{{dataset}}}", ""]
        df.rename(columns=dict(zip(range(len(x_labels)), x_labels)), inplace=True)
        df.set_index('Dataset', inplace=True)
        df_ratios.append(df)
    df_combine = pd.concat(df_ratios, axis=1)
    df_combine.insert(0, 'Method', columns)
    return df_combine


def print_model_diff(datasets, n_trees, remove_ratios):
    summary = []
    for dataset in datasets:
        df_combine = summary_model_diff_for_dataset(dataset, n_trees, remove_ratios)
        df_combine.style.format(lambda x: f"{x:.6f}")
        summary.append('\n'.join(df_combine.style.to_latex().split('\n')[3:5]))
    latex_table = '\n\\midrule\n'.join(summary)
    print(latex_table)


class ModelDiff:
    def __init__(self, dataset, n_trees, remove_ratio, n_rounds):
        self.dataset = dataset
        self.n_trees = n_trees
        self.remove_ratio = remove_ratio
        self.n_rounds = n_rounds

        # load data
        self.remain_X, self.remain_y = load_data(f"../data/{dataset}.train.remain_{remove_ratio}", data_fmt='csv',
                                                 output_dense=True)
        self.delete_X, self.delete_y = load_data(f"../data/{dataset}.train.delete_{remove_ratio}", data_fmt='csv',
                                                 output_dense=True)
        self.test_X, self.test_y = load_data(f"../data/{dataset}.test", data_fmt='csv', output_dense=True)

        self.original_score_on_remain = np.zeros((n_rounds, self.remain_X.shape[0]))
        self.retrain_score_on_remain = np.zeros((n_rounds, self.remain_X.shape[0]))
        self.deleted_score_on_remain = np.zeros((n_rounds, self.remain_X.shape[0]))
        self.original_score_on_delete = np.zeros((n_rounds, self.delete_X.shape[0]))
        self.retrain_score_on_delete = np.zeros((n_rounds, self.delete_X.shape[0]))
        self.deleted_score_on_delete = np.zeros((n_rounds, self.delete_X.shape[0]))
        self.original_score_on_test = np.zeros((n_rounds, self.test_X.shape[0]))
        self.retrain_score_on_test = np.zeros((n_rounds, self.test_X.shape[0]))
        self.deleted_score_on_test = np.zeros((n_rounds, self.test_X.shape[0]))

    def predict_(self, n_used_trees=None):
        for i in range(self.n_rounds):
            # define path of ith model
            model_path = f"../cache/{self.dataset}_tree{self.n_trees}_original_{self.remove_ratio}_{i}_deltaboost.json"
            model_remain_path = f"../cache/{self.dataset}_tree{self.n_trees}_retrain_{self.remove_ratio}_{i}_deltaboost.json"
            model_deleted_path = f"../cache/{self.dataset}_tree{self.n_trees}_original_{self.remove_ratio}_{i}_deleted.json"

            # load model
            with open(model_path, 'r') as f1, open(model_remain_path, 'r') as f2, open(model_deleted_path, 'r') as f3:
                js1 = json.load(f1)
                js2 = json.load(f2)
                js3 = json.load(f3)
            gbdt_original = GBDT.load_from_json(js1)
            gbdt_retrain = GBDT.load_from_json(js2)
            gbdt_deleted = GBDT.load_from_json(js3)

            # predict and append
            self.original_score_on_remain[i] = gbdt_original.predict_score(self.remain_X, n_used_trees)
            self.retrain_score_on_remain[i] = gbdt_retrain.predict_score(self.remain_X, n_used_trees)
            self.deleted_score_on_remain[i] = gbdt_deleted.predict_score(self.remain_X, n_used_trees)
            self.original_score_on_delete[i] = gbdt_original.predict_score(self.delete_X, n_used_trees)
            self.retrain_score_on_delete[i] = gbdt_retrain.predict_score(self.delete_X, n_used_trees)
            self.deleted_score_on_delete[i] = gbdt_deleted.predict_score(self.delete_X, n_used_trees)
            self.original_score_on_test[i] = gbdt_original.predict_score(self.test_X, n_used_trees)
            self.retrain_score_on_test[i] = gbdt_retrain.predict_score(self.test_X, n_used_trees)
            self.deleted_score_on_test[i] = gbdt_deleted.predict_score(self.test_X, n_used_trees)

    def get_mean_diff(self, n_used_trees=None, average=True):
        """
        Get the mean difference between retrain & delete on remain, delete, test;
        Get the mean difference between original & delete on remain, delete, test;
        :return:
        """
        # mean by rounds
        retrain_vs_delete_on_remain = np.mean(self.retrain_score_on_remain - self.deleted_score_on_remain, axis=0)
        original_vs_retrain_on_remain = np.mean(self.original_score_on_remain - self.retrain_score_on_remain, axis=0)
        retrain_vs_delete_on_delete = np.mean(self.retrain_score_on_delete - self.deleted_score_on_delete, axis=0)
        original_vs_retrain_on_delete = np.mean(self.original_score_on_delete - self.retrain_score_on_delete, axis=0)
        retrain_vs_delete_on_test = np.mean(self.retrain_score_on_test - self.deleted_score_on_test, axis=0)
        original_vs_retrain_on_test = np.mean(self.original_score_on_test - self.retrain_score_on_test, axis=0)

        if average:
            # Average of square of the difference by samples
            retrain_vs_delete_on_remain_avg = np.sqrt(np.mean(retrain_vs_delete_on_remain ** 2))
            original_vs_retrain_on_remain_avg = np.sqrt(np.mean(original_vs_retrain_on_remain ** 2))
            retrain_vs_delete_on_delete_avg = np.sqrt(np.mean(retrain_vs_delete_on_delete ** 2))
            original_vs_retrain_on_delete_avg = np.sqrt(np.mean(original_vs_retrain_on_delete ** 2))
            retrain_vs_delete_on_test_avg = np.sqrt(np.mean(retrain_vs_delete_on_test ** 2))
            original_vs_retrain_on_test_avg = np.sqrt(np.mean(original_vs_retrain_on_test ** 2))

            return retrain_vs_delete_on_remain_avg, original_vs_retrain_on_remain_avg, \
                   retrain_vs_delete_on_delete_avg, original_vs_retrain_on_delete_avg, \
                   retrain_vs_delete_on_test_avg, original_vs_retrain_on_test_avg
        else:
            return retrain_vs_delete_on_remain, original_vs_retrain_on_remain, \
                   retrain_vs_delete_on_delete, original_vs_retrain_on_delete, \
                   retrain_vs_delete_on_test, original_vs_retrain_on_test

    def get_hellinger_distance(self, n_used_trees=None, n_bins=50):
        # # get the mean difference (scale)
        # retrain_vs_delete_on_remain_mean, original_vs_retrain_on_remain_mean, \
        # retrain_vs_delete_on_delete_mean, original_vs_retrain_on_delete_mean, \
        # retrain_vs_delete_on_test_mean, original_vs_retrain_on_test_mean = self.get_mean_diff(n_used_trees,
        #                                                                                       average=False)

        # cluster self.retrain_score_on_delete into histogram
        min_value = min(np.min(self.retrain_score_on_delete), np.min(self.deleted_score_on_delete),
                        np.min(self.retrain_score_on_remain), np.min(self.deleted_score_on_remain),
                        np.min(self.retrain_score_on_test), np.min(self.deleted_score_on_test))
        max_value = max(np.max(self.retrain_score_on_delete), np.max(self.deleted_score_on_delete),
                        np.max(self.retrain_score_on_remain), np.max(self.deleted_score_on_remain),
                        np.max(self.retrain_score_on_test), np.max(self.deleted_score_on_test))

        n_instances_remain = self.remain_X.shape[0]
        original_on_remain_hist = np.zeros((n_instances_remain, n_bins))
        retrain_on_remain_hist = np.zeros((n_instances_remain, n_bins))
        delete_on_remain_hist = np.zeros((n_instances_remain, n_bins))
        for i in range(n_instances_remain):
            original_on_remain_hist[i, :], _ = np.histogram(self.original_score_on_remain[:, i], bins=n_bins,
                                                            range=(min_value, max_value), density=True)
            retrain_on_remain_hist[i, :], _ = np.histogram(self.retrain_score_on_remain[:, i], bins=n_bins,
                                                           range=(min_value, max_value), density=True)
            delete_on_remain_hist[i, :], _ = np.histogram(self.deleted_score_on_remain[:, i], bins=n_bins,
                                                           range=(min_value, max_value), density=True)

        n_instances_delete = self.delete_X.shape[0]
        original_on_delete_hist = np.zeros((n_instances_delete, n_bins))
        retrain_on_delete_hist = np.zeros((n_instances_delete, n_bins))
        delete_on_delete_hist = np.zeros((n_instances_delete, n_bins))
        for i in range(n_instances_delete):
            original_on_delete_hist[i, :], _ = np.histogram(self.original_score_on_delete[:, i], bins=n_bins,
                                                            range=(min_value, max_value), density=True)
            retrain_on_delete_hist[i, :], _ = np.histogram(self.retrain_score_on_delete[:, i], bins=n_bins,
                                                           range=(min_value, max_value), density=True)
            delete_on_delete_hist[i, :], _ = np.histogram(self.deleted_score_on_delete[:, i], bins=n_bins,
                                                           range=(min_value, max_value), density=True)

        n_instances_test = self.test_X.shape[0]
        original_on_test_hist = np.zeros((n_instances_test, n_bins))
        retrain_on_test_hist = np.zeros((n_instances_test, n_bins))
        delete_on_test_hist = np.zeros((n_instances_test, n_bins))
        for i in range(n_instances_test):
            original_on_test_hist[i, :], _ = np.histogram(self.original_score_on_test[:, i], bins=n_bins,
                                                          range=(min_value, max_value), density=True)
            retrain_on_test_hist[i, :], _ = np.histogram(self.retrain_score_on_test[:, i], bins=n_bins,
                                                         range=(min_value, max_value), density=True)
            delete_on_test_hist[i, :], _ = np.histogram(self.deleted_score_on_test[:, i], bins=n_bins,
                                                         range=(min_value, max_value), density=True)

        # Hellinger distance
        bin_width = (max_value - min_value) / n_bins
        retrain_vs_delete_on_remain = 1 - bin_width * np.sum(
            np.sqrt(retrain_on_remain_hist * delete_on_remain_hist), axis=1)
        original_vs_retrain_on_remain = 1 - bin_width * np.sum(
            np.sqrt(original_on_remain_hist * retrain_on_remain_hist), axis=1)
        retrain_vs_delete_on_delete = 1 - bin_width * np.sum(
            np.sqrt(retrain_on_delete_hist * delete_on_delete_hist), axis=1)
        original_vs_retrain_on_delete = 1 - bin_width * np.sum(
            np.sqrt(original_on_delete_hist * retrain_on_delete_hist), axis=1)
        retrain_vs_delete_on_test = 1 - bin_width * np.sum(
            np.sqrt(retrain_on_test_hist * delete_on_test_hist), axis=1)
        original_vs_retrain_on_test = 1 - bin_width * np.sum(
            np.sqrt(original_on_test_hist * retrain_on_test_hist), axis=1)

        # get the average of Hellinger distance
        retrain_vs_delete_on_remain_mean = np.mean(retrain_vs_delete_on_remain)
        original_vs_retrain_on_remain_mean = np.mean(original_vs_retrain_on_remain)
        retrain_vs_delete_on_delete_mean = np.mean(retrain_vs_delete_on_delete)
        original_vs_retrain_on_delete_mean = np.mean(original_vs_retrain_on_delete)
        retrain_vs_delete_on_test_mean = np.mean(retrain_vs_delete_on_test)
        original_vs_retrain_on_test_mean = np.mean(original_vs_retrain_on_test)

        return retrain_vs_delete_on_remain_mean, original_vs_retrain_on_remain_mean, \
                retrain_vs_delete_on_delete_mean, original_vs_retrain_on_delete_mean, \
                retrain_vs_delete_on_test_mean, original_vs_retrain_on_test_mean

    def print_mean_diff(self, n_used_trees=None):
        retrain_vs_delete_on_remain_avg, original_vs_retrain_on_remain_avg, \
        retrain_vs_delete_on_delete_avg, original_vs_retrain_on_delete_avg, \
        retrain_vs_delete_on_test_avg, original_vs_retrain_on_test_avg = self.get_mean_diff(n_used_trees)
        x_labels = [r'$D_f$', r'$D_r$', r'$D_{test}$']
        df = pd.DataFrame(data={
            'original vs. retrain': [original_vs_retrain_on_remain_avg, original_vs_retrain_on_delete_avg,
                                     original_vs_retrain_on_test_avg],
            'retrain vs. delete': [retrain_vs_delete_on_remain_avg, retrain_vs_delete_on_delete_avg,
                                   retrain_vs_delete_on_test_avg]
        }, index=x_labels)
        df = df.T
        df['Dataset'] = [f"\\multirow{{2}}{{*}}{{{self.dataset}}}", ""]
        df.rename(columns=dict(zip(range(len(x_labels)), x_labels)), inplace=True)
        df.set_index('Dataset', inplace=True)
        df.style.format(lambda x: f"{x:.6f}")
        print(df.to_latex(escape=False))

    def get_helliger_distance_df(self, n_used_trees=None):
        retrain_vs_delete_on_remain, original_vs_retrain_on_remain, \
        retrain_vs_delete_on_delete, original_vs_retrain_on_delete, \
        retrain_vs_delete_on_test, original_vs_retrain_on_test = self.get_hellinger_distance(n_used_trees)
        x_labels = [r'$D_f$', r'$D_r$', r'$D_{test}$']
        df = pd.DataFrame(data={
            'original vs. retrain': [original_vs_retrain_on_remain, original_vs_retrain_on_delete,
                                     original_vs_retrain_on_test],
            'retrain vs. delete': [retrain_vs_delete_on_remain, retrain_vs_delete_on_delete,
                                   retrain_vs_delete_on_test]
        }, index=x_labels)
        columns = df.columns
        df = df.T
        df['Dataset'] = [f"\\multirow{{2}}{{*}}{{{self.dataset}}}", ""]
        df.rename(columns=dict(zip(range(len(x_labels)), x_labels)), inplace=True)
        df.set_index('Dataset', inplace=True)
        df.style.format(lambda x: f"{x:.6f}")
        df.insert(0, 'Helliger distance', columns)
        return df

    def print_helliger_distance(self, n_used_trees=None):
        retrain_vs_delete_on_remain_avg, original_vs_retrain_on_remain_avg, \
        retrain_vs_delete_on_delete_avg, original_vs_retrain_on_delete_avg, \
        retrain_vs_delete_on_test_avg, original_vs_retrain_on_test_avg = self.get_hellinger_distance(n_used_trees)
        x_labels = [r'$D_f$', r'$D_r$', r'$D_{test}$']
        df = pd.DataFrame(data={
            'original vs. retrain': [original_vs_retrain_on_remain_avg, original_vs_retrain_on_delete_avg,
                                     original_vs_retrain_on_test_avg],
            'retrain vs. delete': [retrain_vs_delete_on_remain_avg, retrain_vs_delete_on_delete_avg,
                                   retrain_vs_delete_on_test_avg]
        }, index=x_labels)
        df = df.T
        df['Dataset'] = [f"\\multirow{{2}}{{*}}{{{self.dataset}}}", ""]
        df.rename(columns=dict(zip(range(len(x_labels)), x_labels)), inplace=True)
        df.set_index('Dataset', inplace=True)
        df.style.format(lambda x: f"{x:.6f}")
        print(df.to_latex(escape=False))


if __name__ == '__main__':
    datasets = ['codrna', 'cadata', 'covtype', 'gisette', 'msd']
    # datasets = ['cadata']
    remove_ratios = ['1e-03', '1e-02']
    # remove_ratios = ['0.001', '0.01']
    # plot_score_before_after_removal("../out/remove_test/tree50", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree30", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree10", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree1", datasets, remove_ratios)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=50)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=30)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=10)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=1)
    # print_model_diff(datasets, 1, remove_ratios)
    # print_model_diff(datasets, 10, remove_ratios)
    # print_model_diff(datasets, 30, remove_ratios)
    # print_model_diff(datasets, 50, remove_ratios)

    model_diff1 = ModelDiff('cadata', 1, '1e-03', 10)
    model_diff1.predict_(1)
    df1 = model_diff1.get_helliger_distance_df(1)
    model_diff2 = ModelDiff('cadata', 1, '1e-02', 10)
    model_diff2.predict_(1)
    df2 = model_diff2.get_helliger_distance_df(1)
    df_combine = pd.concat([df1, df2.drop(columns=df2.columns[0])], axis=1)
    print(df_combine.to_latex(escape=False))

    model_diff1 = ModelDiff('codrna', 1, '1e-03', 10)
    model_diff1.predict_(1)
    df1 = model_diff1.get_helliger_distance_df(1)
    model_diff2 = ModelDiff('codrna', 1, '1e-02', 10)
    model_diff2.predict_(1)
    df2 = model_diff2.get_helliger_distance_df(1)
    df_combine = pd.concat([df1, df2.drop(columns=df2.columns[0])], axis=1)
    print(df_combine.to_latex(escape=False))
