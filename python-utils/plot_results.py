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
        pattern = re.compile(".*INFO (?:deltaboost|gbdt)\.cpp:[0-9]* :.*(?:error|RMSE) = ([+-]?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)")
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
                ax = df.plot(kind='bar', rot=0, xlabel="Datset partition", ylabel=metric, title=f"{dataset} removing {ratio}")
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
        ratio = 0.01    # either should be the same
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
        remain_X, remain_y = load_data(f"../data/{dataset}.train.remain_{remove_ratio}", data_fmt='csv', output_dense=True)
        delete_X, delete_y = load_data(f"../data/{dataset}.train.delete_{remove_ratio}", data_fmt='csv', output_dense=True)
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
        df = pd.DataFrame(data={'original vs. retrain': [diff_retrain_original_on_remain, diff_retrain_original_on_delete, diff_retrain_original_on_test],
                                'delete vs. retrain': [diff_retrain_delete_on_remain, diff_retrain_delete_on_delete, diff_retrain_delete_on_test]})
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


if __name__ == '__main__':
    # datasets = ['codrna', 'cadata', 'covtype', 'gisette', 'msd']
    datasets = ['cadata']
    remove_ratios = ['1e-03', '1e-02']
    # remove_ratios = ['0.001', '0.01']
    # plot_score_before_after_removal("../out/remove_test/tree50", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree30", datasets, remove_ratios)
    plot_score_before_after_removal("../out/remove_test/tree10", datasets, remove_ratios)
    # plot_score_before_after_removal("../out/remove_test/tree1", datasets, remove_ratios)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=50)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=30)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=10)
    # plot_deltaboost_vs_gbdt("../out/remove_test", datasets, n_trees=1)
    # print_model_diff(datasets, 1, remove_ratios)
    print_model_diff(datasets, 10, remove_ratios)
    # print_model_diff(datasets, 30, remove_ratios)
    # print_model_diff(datasets, 50, remove_ratios)
