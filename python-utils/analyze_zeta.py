import os

import numpy as np
import ujson as json

import matplotlib.pyplot as plt
import pandas as pd

cache_root = "/data/zhaomin/DeltaBoost/cache"
data_root = "/data/zhaomin/DeltaBoost/data"

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

datasets = ['codrna', 'cadata', 'gisette', 'msd', 'susy']
mean_zeta = []
std_zeta = []
for dataset in datasets:
    cache_file_name = f"{dataset}_tree1_original_1e-02_0_deltaboost.json"
    data_file_name = f"{dataset}.train"
    print(f"Loading {cache_file_name}...")
    with open(os.path.join(cache_root, cache_file_name), "r") as f:
        js = json.load(f)
        gh_pairs_raw = js['deltaboost']['gh_pairs_per_sample']
        gh_pairs = np.array([[x['g'], x['h']] for x in gh_pairs_raw[0]])
    print(f"Loading {data_file_name}...")
    with open(os.path.join(data_root, data_file_name), "r") as f:
        data = pd.read_csv(f, header=None, sep=',').to_numpy()
        X = data[:, 1:]

    print(f"Plotting {dataset}...")
    fid = js['deltaboost']['trees'][0][0]['nodes'][0]['split_feature_id']
    x = X[:, fid]
    bin_edges = np.quantile(x, np.linspace(0, 1, 256))
    # fit the histogram to g and h
    g_hist, _ = np.histogram(x, bins=bin_edges, weights=gh_pairs[:, 0])
    h_hist, _ = np.histogram(x, bins=bin_edges, weights=gh_pairs[:, 1])

    # align left and right histograms
    g_hist = np.concatenate([[0], g_hist])
    h_hist = np.concatenate([[0], h_hist])

    a_l = np.cumsum(g_hist ** 2)
    b_l = np.cumsum(g_hist)
    c_l = np.cumsum(h_hist)
    a_r = np.cumsum(g_hist[::-1] ** 2)[::-1]
    b_r = np.cumsum(g_hist[::-1])[::-1]
    c_r = np.cumsum(h_hist[::-1])[::-1]

    zeta_up = a_r * c_l ** 2 + a_l * c_r ** 2
    zeta_down = (b_l * c_r - b_r * c_l) ** 2 + 1e-10
    zeta = zeta_up[1:-1] / zeta_down[1:-1]

    remove_ratio = 0.01
    scaled_zeta = zeta * remove_ratio

    plt.plot(bin_edges[1:-1], scaled_zeta)
    plt.show()

    mean_zeta.append(np.mean(scaled_zeta))
    std_zeta.append(np.std(scaled_zeta))
    pass

print(" & ".join(datasets))
print(" & ".join([fr"{x:.4f} \textpm {y:.4f}" for x, y in zip(mean_zeta, std_zeta)]))


