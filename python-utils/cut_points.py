import numpy as np
from train_test_split import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multinomial
from GBDT import Tree, Node
import json


def check_vulnerability(n_samples_in_bins, bid_to_nid, split_info: Tree, n_removing_samples, threshold, n_rounds=10000):
    n_removed_samples_in_bin_rv = multinomial(n=n_removing_samples, p=n_samples_in_bins / np.sum(n_samples_in_bins))
    n_removed_samples_in_bin_rvs = n_removed_samples_in_bin_rv.rvs(n_rounds)

    # update number of indices in each node
    for nid, r in zip(bid_to_nid, n_removed_samples_in_bin_rvs.T):
        node = split_info.nodes[nid]
        node.split_value = - r + node.split_value  # after deletion
        node.is_leaf = True
        while node.parent_id != -1:
            node = split_info.nodes[node.parent_id]
            node.split_value = - r + node.split_value

    # check if tree has vulnerable nodes (internal nodes with n_samples < threshold)
    visit = [0]
    n_vulnerable = np.zeros([n_rounds, 0])
    while len(visit) > 0:
        nid = visit.pop(0)
        node = split_info.nodes[nid]
        is_vulnerable = (node.split_value < threshold) & (not node.is_leaf)  # split_feature_id = is_vulnerable
        # if np.count_nonzero(is_vulnerable) > 0:
        #     print("")
        n_vulnerable = np.concatenate([n_vulnerable, is_vulnerable.reshape(-1, 1)], axis=1)

        if not node.is_leaf:
            visit.append(node.lch_id)
            visit.append(node.rch_id)
    mean_vulnerable = np.mean(np.count_nonzero(n_vulnerable, axis=1))
    std_vulnerable = np.std(np.count_nonzero(n_vulnerable, axis=1))
    return mean_vulnerable, std_vulnerable


def load_cut_points(raw_feature_value, threshold, tol=1e-8):
    feature_value = np.sort(raw_feature_value.flatten())
    min_value = feature_value[0]
    max_value = feature_value[-1] + 0.5
    splittable = np.array([True])
    n_samples_in_bins = np.array([feature_value.shape[0]])
    splits = np.array([min_value, max_value])
    # split_info = Tree()
    # split_info.add_root_(Node(split_value=feature_value.shape[0]))  # the value is the number of instances
    # bid_to_nid = [0]
    while True:
        split_bin_id = np.argmax(n_samples_in_bins * splittable)  # add abs of min to ensure all values are positive
        if n_samples_in_bins[split_bin_id] < threshold:
            break
        mid_value = (splits[split_bin_id] + splits[split_bin_id + 1]) / 2
        new_samples_in_left_bin = feature_value[(splits[split_bin_id] <= feature_value) & (feature_value < mid_value)]
        new_samples_in_right_bin = feature_value[
            (mid_value <= feature_value) & (feature_value < splits[split_bin_id + 1])]
        left_splittable = ~np.isclose(splits[split_bin_id], mid_value, atol=tol)
        right_splittable = ~np.isclose(splits[split_bin_id + 1], mid_value, atol=tol)
        splits = np.insert(splits, split_bin_id + 1, mid_value)
        n_samples_in_bins[split_bin_id] = new_samples_in_left_bin.shape[0]
        n_samples_in_bins = np.insert(n_samples_in_bins, split_bin_id + 1, new_samples_in_right_bin.shape[0])
        splittable[split_bin_id] = left_splittable
        splittable = np.insert(splittable, split_bin_id, right_splittable)

        # left_node = Node(split_value=n_samples_in_bins[split_bin_id], parent_id=bid_to_nid[split_bin_id])
        # right_node = Node(split_value=n_samples_in_bins[split_bin_id + 1], parent_id=bid_to_nid[split_bin_id])
        # split_info.add_child_(left_node, is_right=False)
        # bid_to_nid[split_bin_id] = len(split_info.nodes) - 1
        # split_info.add_child_(right_node, is_right=True)
        # bid_to_nid = np.insert(bid_to_nid, split_bin_id + 1, len(split_info.nodes) - 1)

    non_zero_idx = np.array(np.where(n_samples_in_bins > 0))
    n_samples_in_bins = n_samples_in_bins[non_zero_idx].flatten()
    splits = np.insert(splits[non_zero_idx + 1], 0, splits[0])

    return n_samples_in_bins, splits
    # return n_samples_in_bins, splits, bid_to_nid, split_info


def check_split_equal(x, split1, split2):
    sorted_x = np.sort(x)
    split_id1 = sorted_x.searchsorted(split1)
    split_id2 = sorted_x.searchsorted(split2)
    return (np.unique(split_id1) == np.unique(split_id2)).all()


def load_gradients(model, model_type='deltaboost', tree_id=None):
    if isinstance(model, str):
        with open(model, 'r') as f:
            js = json.load(f)
        print("Loaded.")
    elif isinstance(model, dict):
        js = model
    else:
        raise NotImplementedError("Unsupported model type")

    gh_pairs = []
    if tree_id is None:
        for gh_pairs_json in js[model_type]['gh_pairs_per_sample']:
            # half of the vec is zero for unknown reason, just remove, fix later
            gh_pairs_json = gh_pairs_json[:len(gh_pairs_json)]
            gh_pairs_per_tree = np.array([[float(gh['g']), float(gh['h'])] for gh in gh_pairs_json]).T
            gh_pairs.append(gh_pairs_per_tree)
    return np.array(gh_pairs)


def plot_gh_func(X, gh, tree_id=0, feature_id=0, save_path=None):
    indices = np.argsort(-X[:, feature_id])  # argsort in descending order
    gh_for_tree = gh[tree_id, :, indices].T
    cumsum_gh = np.cumsum(gh_for_tree, axis=1)
    plt.plot(cumsum_gh[1], cumsum_gh[0])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def plot_gain_func(X, gh, tree_id=0, feature_id=0, _lambda=1, save_path=None, remove_ratio=None, gh_bin_save_path=None,
                   delta_gain=None):
    if remove_ratio is not None:
        remain_indices = np.random.choice(np.arange(X.shape[0]), size=int(X.shape[0] * (1 - remove_ratio)))
        X = X[remain_indices, :]
        gh = gh[:, :, remain_indices]

    indices = np.argsort(-X[:, feature_id])  # argsort in descending order
    x = X[indices, feature_id]
    gh_for_tree = gh[tree_id, :, indices].T
    n_samples_in_bins, splits = load_cut_points(x, 100)
    split_indices = np.cumsum(n_samples_in_bins)
    gh_for_bins = np.add.reduceat(gh_for_tree, split_indices - 1, axis=1)
    gh_leftsum = np.cumsum(gh_for_bins, axis=1)
    sum_gh = np.sum(gh_for_bins, axis=1)
    gain = np.maximum(gh_leftsum[0] ** 2 / (_lambda + gh_leftsum[1]) +
                      (sum_gh[0] - gh_leftsum[0]) ** 2 / (_lambda + sum_gh[1] - gh_leftsum[1]) -
                      sum_gh[0] ** 2 / (_lambda + sum_gh[1]), 0)
    plt.plot(gh_leftsum[1], gain, label='Gain')
    if delta_gain is not None:
        plt.plot(gh_leftsum[1], gain + delta_gain, label='Gain after removal')
        plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    if gh_bin_save_path is not None:
        plt.plot(gh_leftsum[1], gh_leftsum[0])
        plt.savefig(gh_bin_save_path)
    plt.close()


def analyze_delta_gain(X, gh, tree_id=0, feature_id=0, _lambda=1, remove_ratio=0.01, n_rounds=1000,
                       delta_gain_img_path=None):
    indices = np.argsort(-X[:, feature_id])  # argsort in descending order
    x = X[indices, feature_id]
    gh_for_tree = gh[tree_id, :, indices].T
    n_samples_in_bins, splits = load_cut_points(x, 100)
    split_indices = np.cumsum(n_samples_in_bins)
    gh_for_bins = np.add.reduceat(gh_for_tree, split_indices - 1, axis=1)
    gh_leftsum = np.cumsum(gh_for_bins, axis=1)
    sum_gh = np.sum(gh_for_bins, axis=1)
    gain = np.maximum(gh_leftsum[0] ** 2 / (_lambda + gh_leftsum[1]) +
                      (sum_gh[0] - gh_leftsum[0]) ** 2 / (_lambda + sum_gh[1] - gh_leftsum[1]) -
                      sum_gh[0] ** 2 / (_lambda + sum_gh[1]), 0)

    # remove
    remove_indices = np.array([np.random.choice(np.arange(X.shape[0]), size=int(X.shape[0] * remove_ratio))
                               for _ in range(n_rounds)])
    remove_gh_for_tree = np.transpose(gh[tree_id, :, remove_indices], [0, 2, 1])
    remove_gh_for_bins = np.zeros([remove_gh_for_tree.shape[0], gh_for_bins.shape[0], gh_for_bins.shape[1]])
    for i, (remove_gh, remove_id) in enumerate(zip(remove_gh_for_tree, remove_indices)):
        bin_id = np.searchsorted(split_indices, remove_id, side='right')
        remove_gh_for_bins[i][:, bin_id] += remove_gh
    remain_gh_for_bins = gh_for_bins - remove_gh_for_bins
    remain_gh_leftsum = np.cumsum(remain_gh_for_bins, axis=2)
    remain_sum_gh = np.sum(remain_gh_for_bins, axis=2)
    remain_gain = np.maximum(remain_gh_leftsum[:, 0, :] ** 2 / (_lambda + remain_gh_leftsum[:, 1, :]) +
                             (remain_sum_gh[:, 0].reshape(-1, 1) - remain_gh_leftsum[:, 0, :]) ** 2 /
                             (_lambda + remain_sum_gh[:, 1].reshape(-1, 1) - remain_gh_leftsum[:, 1, :]) -
                             remain_sum_gh[:, 0].reshape(-1, 1) ** 2 / (_lambda + remain_sum_gh[:, 1].reshape(-1, 1)), 0)
    mean_delta_gain = np.mean(remain_gain - gain, axis=0)
    std_delta_gain = np.std(remain_gain - gain, axis=0)
    mean_remain_gain = np.mean(remain_gain, axis=0)
    std_remain_gain = np.std(remain_gain, axis=0)

    if delta_gain_img_path is not None:
        plt.errorbar(gh_leftsum[1], mean_delta_gain, std_delta_gain * 3)
        # plt.plot(gh_leftsum[1], gain, 'r')
        plt.savefig(delta_gain_img_path)
        plt.close()

    return mean_delta_gain, std_delta_gain

    # n_remove = np.ceil(gh.shape[2] * remove_ratio).astype('int')
    # for tree_id, gh_for_tree in enumerate(gh):
    #     top_g = np.sort(gh_for_tree[0])[-n_remove:]
    #     worst_delta_gain = np.sum(top_g) ** 2 * 3
    #     random_g = np.array([np.random.choice(gh_for_tree[0], n_remove, replace=False) for _ in range(n_rounds)])
    #     random_gain = np.sum(random_g, axis=1) ** 2 * 3
    #     delta_gain_mean = np.mean(random_gain)
    #     delta_gain_std = np.std(random_gain)
    #     print(f"Tree {tree_id}: {worst_delta_gain=}, {delta_gain_mean=}, {delta_gain_std=}")


if __name__ == '__main__':
    X, y = load_data("../data/codrna.train", data_fmt='libsvm', output_dense=True)
    gh = load_gradients("../cache/codrna.json")
    # gh = np.array([np.ones(X.shape[0]), np.ones(X.shape[0]) / 4]).reshape([1, 2, -1])
    # plot_gain_func(X, gh, tree_id=0, feature_id=2)
    for tree_id in range(50):
        for feature_id in range(8):
            # plot_gh_func(X, gh, tree_id=tree_id, feature_id=feature_id,
            #              save_path=f"fig/gh_func/codrna/gh_tree_{tree_id}_feature_{feature_id}.jpg")
            # plot_gain_func(X, gh, tree_id=tree_id, feature_id=feature_id,
            #                save_path=f"fig/gain_func/codrna/gain_tree_{tree_id}_feature_{feature_id}.jpg",
            #                gh_bin_save_path=f"fig/gh_func/codrna/gh_tree_{tree_id}_feature_{feature_id}.jpg")
            # plot_gain_func(X, gh, tree_id=tree_id, feature_id=feature_id, remove_ratio=0.01,
            #                save_path=f"fig/gain_func_remove_0.01/codrna/gain_tree_{tree_id}_feature_{feature_id}.jpg",
            #                gh_bin_save_path=f"fig/gh_func_remove_0.01/codrna/gh_tree_{tree_id}_feature_{feature_id}.jpg")
            mean, std = analyze_delta_gain(X, gh, tree_id=tree_id, feature_id=feature_id,
                                           delta_gain_img_path=f"fig/delta_gain/codrna/gh_tree_{tree_id}_feature_{feature_id}.jpg")
        print(f"Tree {tree_id} done.")

    # threshold = 5000
    # n_removing_samples = int(X.shape[0] * 0.01)
    # for x in X.T:
    #     x_remain = np.random.choice(x, size=[x.shape[0] - n_removing_samples], replace=False)
    #     n_samples_in_bins, splits, bid_to_nid, split_info = load_cut_points(x, threshold)
    #     n_samples_in_bins_remain, splits_remain, _, _ = load_cut_points(x_remain, threshold)
    #     mean, std = check_vulnerability(n_samples_in_bins, bid_to_nid, split_info, n_removing_samples, threshold)
    #     print(f"Diff mean: {mean}, std: {std}, num_bins {len(n_samples_in_bins)}")
    #     pass
    #     assert (np.intersect1d(splits, splits_remain) == splits_remain).all()
    #     assert splits.shape[0] == splits_remain.shape[0] and (splits == splits_remain).all()
    #     assert check_split_equal(x_remain, splits, splits_remain)
    #
    #     fig, ax = plt.subplots(3, 1)
    #     feature_hist, _ = np.histogram(x, bins=n_samples_in_bins.shape[0], density=False)
    #     feature_hist = feature_hist[feature_hist > 0].flatten()
    #     ax[2].hist(feature_hist, bins=30)
    #     large_x_lim = ax[2].get_xlim()
    #     flat_hist = np.ones_like(n_samples_in_bins) * x.shape[0] // n_samples_in_bins.shape[0]
    #     ax[0].hist(flat_hist, bins=30, range=large_x_lim)
    #     ax[0].set_xlim(large_x_lim)
    #     ax[1].hist(n_samples_in_bins, bins=30)
    #     ax[1].set_xlim(large_x_lim)
    #     ax[0].set_xlabel("Number of instances in bin (instance-based)")
    #     ax[1].set_xlabel("Number of instances in bin (proposed)")
    #     ax[2].set_xlabel("Number of instances in bin (naive feature-based)")
    #     ax[1].set_ylabel("Number of bins")
    #     plt.tight_layout()
    #     plt.show()
