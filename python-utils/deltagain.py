import os
import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

from cut_points import load_cut_points, load_gradients
from train_test_split import load_data


class ElementarySymmetricFunction:
    def __init__(self, S: np.ndarray):
        """
        :param S: 1d numpy array
        """
        self.S = S.flatten()
        self.esp_mtx = None

        self.calc_esp_()

    def calc_esp_(self):
        """
        calculate esp by dynamic programming, time complexity O(|S|^2)
        :return:
        """
        S = self.S      # for brief
        self.esp_mtx = np.zeros([S.size + 1, S.size + 1])
        self.esp_mtx[0, :] = np.ones([S.size + 1])

        for i in range(1, self.esp_mtx.shape[0]):    # k
            for j in range(1, self.esp_mtx.shape[1]):     # n
                self.esp_mtx[i, j] = self.esp_mtx[i, j - 1] + S[j - 1] * self.esp_mtx[i - 1, j - 1]

    def esp(self, k):
        return self.esp_mtx[k, self.S.size]


class Hist:
    def __init__(self, split_indices, splits, gh, gh_in_bins, g_square_in_bins=None):
        self.split_indices = split_indices
        self.splits = splits
        self.gh = gh
        self.gh_in_bins = gh_in_bins
        self.g_square_in_bins = g_square_in_bins

        if not self._check_size():
            raise AssertionError(f"Sizes of inputs unmatched, got {self.split_indices.shape[0]=}, "
                                 f"{self.splits.shape[0]=}, "
                                 f"{self.gh_in_bins.shape[1]=}")

    @property
    def n_instances(self):
        return self.gh.shape[1]

    def _check_size(self):
        return self.splits.shape[0] - 1 == self.split_indices.shape[0] == self.gh_in_bins.shape[1]

    def get_gain(self, _lambda):
        gh_leftsum = np.cumsum(self.gh_in_bins, axis=1)
        sum_gh = np.sum(self.gh_in_bins, axis=1)
        gain = np.maximum(gh_leftsum[0] ** 2 / (_lambda + gh_leftsum[1]) +
                          (sum_gh[0] - gh_leftsum[0]) ** 2 / (_lambda + sum_gh[1] - gh_leftsum[1]) -
                          sum_gh[0] ** 2 / (_lambda + sum_gh[1]), -np.inf)
        return gain

    def remove_ratio_to_n(self, remove_ratio):
        return int(self.gh.shape[1] * remove_ratio)

    @classmethod
    def generate_robust_hist(cls, x, gh, threshold=100):
        indices = np.argsort(-x)
        x = x[indices]
        gh = gh[:, indices]
        n_samples_in_bins, splits = load_cut_points(x, threshold=threshold)
        split_indices = np.zeros(n_samples_in_bins.shape, dtype='int')
        split_indices[1:] = np.cumsum(n_samples_in_bins)[:-1]   # set first id as 0, remove the last id
        gh_in_bins = np.add.reduceat(gh, split_indices, axis=1)
        g_square_in_bins = np.add.reduceat(gh[0] ** 2, split_indices)
        return cls(split_indices, splits, gh, gh_in_bins, g_square_in_bins)

    @staticmethod
    def probs(n, n_d, i, b):
        coef1 = comb(b, i)
        coef2 = np.prod(np.arange(n_d-i+1, n_d+1))
        divisor = np.arange(n-n_d+1, n+1)
        dividend = np.ones([n_d])
        divisor[:n_d-i] = np.arange(n-b-n_d+i+1, n-b+1)
        coef3 = dividend / divisor
        probs = coef1 * coef2 * coef3
        # assert (0 <= probs).all() and (probs <= 1).all()
        return probs

    def calc_ev_removed_gradients_in_bins(self, remove_ratio):
        return self.gh_in_bins[0] * remove_ratio

    def calc_ev_removed_gradients_square_in_bins(self, remove_ratio):
        n_remove = self.remove_ratio_to_n(remove_ratio)
        ev = remove_ratio * (self.g_square_in_bins + (n_remove - 1) / self.n_instances * self.gh_in_bins[0] ** 2)
        return ev

    def calc_ev_second_moment_leftsum(self, remove_ratio):
        n_remove = self.remove_ratio_to_n(remove_ratio)
        g_square_leftsum = np.cumsum(self.g_square_in_bins)
        g_leftsum = np.cumsum(self.gh_in_bins[0])
        ev = remove_ratio * (g_square_leftsum + (n_remove - 1) / self.n_instances * (g_leftsum ** 2))
        return ev

    def calc_ev_delta_gain_in_bins(self, remove_ratio, _lambda=1):
        n_remove = self.remove_ratio_to_n(remove_ratio)
        ev_remove_gradients_in_bins = self.calc_ev_removed_gradients_in_bins(remove_ratio)
        ev_remove_gradients_sum = np.sum(ev_remove_gradients_in_bins)
        ev_remove_gradients_leftsum = np.cumsum(ev_remove_gradients_in_bins)
        ev_remove_gradients_rightsum = ev_remove_gradients_sum - ev_remove_gradients_leftsum
        g_square_sum = np.sum(self.g_square_in_bins)
        g_square_leftsum = np.cumsum(self.g_square_in_bins)
        g_square_rightsum = g_square_sum - g_square_leftsum
        gh_sum = np.sum(self.gh_in_bins, axis=1)
        gh_leftsum = np.cumsum(self.gh_in_bins, axis=1)
        gh_rightsum = gh_sum.reshape(-1, 1) - gh_leftsum
        ev_second_moment_left = remove_ratio * (g_square_leftsum + (n_remove - 1) / self.n_instances * gh_leftsum[0] ** 2)
        ev_second_moment_right = remove_ratio * (g_square_rightsum + (n_remove - 1) / self.n_instances * gh_rightsum[0] ** 2)
        ev_second_moment_all = remove_ratio * (g_square_sum + (n_remove - 1) / self.n_instances * gh_sum[0] ** 2)

        c1 = n_remove / self.n_instances
        c2 = ((self.n_instances - n_remove) ** 2 - n_remove) / self.n_instances ** 2
        ev_remain_gain = (c1 * g_square_leftsum + c2 * gh_leftsum[0] ** 2) / ((1 - c1) * gh_leftsum[1] + _lambda) + \
                        (c1 * g_square_rightsum + c2 * gh_rightsum[0] ** 2) / ((1 - c1) * gh_rightsum[1] + _lambda) - \
                        (c1 * g_square_sum + c2 * gh_sum[0] ** 2) / ((1 - c1) * gh_sum[1] + _lambda)
        ev_delta_gain = ev_remain_gain - hist.get_gain(_lambda)

        ev_delta_gain_lower_bound = (ev_second_moment_left - 2 * gh_leftsum[0] * ev_remove_gradients_leftsum) / \
                                    (gh_leftsum[1] + _lambda) + \
                                    (ev_second_moment_right - 2 * gh_rightsum[0] * ev_remove_gradients_rightsum) / \
                                    (gh_rightsum[1] + _lambda) - \
                                    (ev_second_moment_all - 2 * gh_sum[0] * ev_remove_gradients_sum) / \
                                    (gh_sum[1] + _lambda)

        return ev_delta_gain

    def estimate_removed_gh_in_bins(self, remove_ratio, n_est_rounds=1000, seed=0):
        np.random.seed(seed)
        remove_indices = np.array([np.random.choice(np.arange(self.gh.shape[1]),
                                                    size=self.remove_ratio_to_n(remove_ratio), replace=False)
                                   for _ in range(n_est_rounds)])
        remove_gh = np.transpose(self.gh[:, remove_indices], [1, 0, 2])
        remove_gh_in_bins = np.zeros([remove_gh.shape[0], self.gh_in_bins.shape[0], self.gh_in_bins.shape[1]])
        for i, (rm_gh, remove_id) in enumerate(zip(remove_gh, remove_indices)):
            bin_id = np.searchsorted(self.split_indices, remove_id, side='right') - 1
            rm_g = np.bincount(bin_id, rm_gh[0], remove_gh_in_bins.shape[2])
            rm_h = np.bincount(bin_id, rm_gh[1], remove_gh_in_bins.shape[2])
            remove_gh_in_bins[i] += np.array([rm_g, rm_h])
        return remove_gh_in_bins

    def estimate_delta_gain_in_bins(self, remove_ratio, n_est_rounds=1000, _lambda=1, delta_gain_img_path=None, seed=0):
        remove_gh_in_bins = self.estimate_removed_gh_in_bins(remove_ratio, n_est_rounds, seed=seed)
        remain_gh_in_bins = self.gh_in_bins - remove_gh_in_bins
        remain_gh_leftsum = np.cumsum(remain_gh_in_bins, axis=2)
        remain_gh_sum = np.sum(remain_gh_in_bins, axis=2)
        remain_gain = np.maximum(remain_gh_leftsum[:, 0, :] ** 2 / (_lambda + remain_gh_leftsum[:, 1, :]) +
                                 (remain_gh_sum[:, 0].reshape(-1, 1) - remain_gh_leftsum[:, 0, :]) ** 2 /
                                 (_lambda + remain_gh_sum[:, 1].reshape(-1, 1) - remain_gh_leftsum[:, 1, :]) -
                                 remain_gh_sum[:, 0].reshape(-1, 1) ** 2 / (
                                             _lambda + remain_gh_sum[:, 1].reshape(-1, 1)), -np.inf)
        delta_gain = remain_gain - self.get_gain(_lambda)

        if delta_gain_img_path is not None:
            gh_leftsum = np.cumsum(self.gh_in_bins, axis=1)
            # plt.errorbar(gh_leftsum[1], np.mean(delta_gain, axis=0), np.std(delta_gain, axis=0) * 3)
            plt.plot(gh_leftsum[1], np.mean(delta_gain, axis=0))
            plt.savefig(delta_gain_img_path)
            plt.close()

        return delta_gain


if __name__ == '__main__':
    dataset = "cadata"
    os.makedirs(f"fig/delta_gain/{dataset}", exist_ok=True)
    X, y = load_data(f"../data/{dataset}.train", data_fmt='libsvm', output_dense=True)
    with open(f"../cache/{dataset}.json", 'r') as f:
        js = json.load(f)
    print("Loaded.")
    gh = load_gradients(js)
    calc_time = timedelta()
    est_time = timedelta()
    for tree_id in range(len(js['deltaboost']['trees'])):
        for feature_id in range(X.shape[1]):
            hist = Hist.generate_robust_hist(X[:, feature_id], gh[tree_id])
            time_start = datetime.now()
            ev_delta_gain = hist.calc_ev_delta_gain_in_bins(0.01)
            calc_time_end = datetime.now()
            delta_gain = hist.estimate_delta_gain_in_bins(0.01, 10000)
            est_time_end = datetime.now()
            calc_time += calc_time_end - time_start
            est_time += est_time_end - calc_time_end
            print(f"Tree {tree_id}, feature {feature_id}: " 
                  f"Calculation time {(calc_time_end - time_start).microseconds}, "
                  f"estimation time {(est_time_end - calc_time_end).microseconds}")

            ev_delta_gain_est = np.mean(delta_gain, axis=0)
            gh_leftsum = np.cumsum(hist.gh_in_bins, axis=1)
            plt.plot(gh_leftsum[1], ev_delta_gain_est, label='Estimated')
            plt.plot(gh_leftsum[1], ev_delta_gain, label='Calculated')
            plt.legend()
            plt.ylabel("Expected value of delta-gain")
            plt.xlabel("Accumulated sum of hessians")
            plt.savefig(f"fig/delta_gain/{dataset}/deltagain_ev_tree_{tree_id}_feature_{feature_id}.jpg")
            plt.close()
    print(f"Overall calculation time {calc_time}, "
          f"overall estimation time {est_time}")
