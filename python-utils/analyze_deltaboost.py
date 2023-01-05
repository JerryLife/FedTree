import json
from tqdm import tqdm
import numpy as np

def count_potential_nodes(deltaboost_js: dict, bid_threshold=5, skip_close=False):
    n_potential_nodes_per_tree = []
    n_prior_nodes_per_tree = []
    n_potential_nodes_with_close_split_per_tree = []
    for tree, in tqdm(deltaboost_js['deltaboost']['trees']):
        n_potential_nodes = 0
        n_prior_nodes = 0
        n_potential_nodes_with_close_split = 0
        visiting_indices = [0]
        prior_flags = [True]
        while len(visiting_indices) > 0:
            node_id = visiting_indices.pop(0)
            prior_flag = prior_flags.pop(0)
            node = tree['nodes'][node_id]

            if not node['is_leaf']:
                potential_indices = node['potential_nodes_indices']
                n_potential_nodes += len(potential_indices) - 1
                visiting_indices.append(int(node['lch_index']))
                prior_flags.append(prior_flag)
                visiting_indices.append(int(node['rch_index']))
                prior_flags.append(prior_flag)

                bid_pivots = [node['split_bid']]
                for potential_id in potential_indices[1:]:
                    potential_node = tree['nodes'][potential_id]
                    if not potential_node['is_leaf']:
                        if potential_node['split_feature_id'] == node['split_feature_id'] \
                            and np.isclose(bid_pivots, potential_node['split_bid'], atol=bid_threshold).any():
                            n_potential_nodes_with_close_split += 1
                            if skip_close:
                                continue
                        else:
                            bid_pivots.append(potential_node['split_bid'])
                        visiting_indices.append(potential_node['lch_index'])
                        prior_flags.append(False)
                        visiting_indices.append(potential_node['rch_index'])
                        prior_flags.append(False)

            if node['is_leaf'] or skip_close is False:
                if prior_flag:
                    n_prior_nodes += 1
                else:
                    n_potential_nodes += 1
        n_potential_nodes_per_tree.append(n_potential_nodes)
        n_prior_nodes_per_tree.append(n_prior_nodes)
        n_potential_nodes_with_close_split_per_tree.append(n_potential_nodes_with_close_split)

    return n_prior_nodes_per_tree, n_potential_nodes_per_tree, n_potential_nodes_with_close_split_per_tree


if __name__ == '__main__':
    model_path = "../_cache/codrna.json"
    with open(model_path, 'r') as f:
        js = json.load(f)

    n_prior_nodes_skip, n_potential_nodes_skip, _ = count_potential_nodes(js, 10, True)
    n_prior_nodes, n_potential_nodes, n_potential_nodes_with_close_split = count_potential_nodes(js, 10)
    print(f"Number of prior nodes: {sum(n_prior_nodes)}")
    print(f"Number of potential_nodes: {sum(n_potential_nodes)}")
    print(f"Number of potential nodes with close splits: {sum(n_potential_nodes_with_close_split)}")
    print(f"Ratio of potential nodes with close splits: {sum(n_potential_nodes_with_close_split) / sum(n_potential_nodes)}")
    print(f"{n_prior_nodes=}")
    print(f"{n_potential_nodes=}")
    print(f"{n_potential_nodes_with_close_split=}")
    print(f"Ratio of skipped nodes {1 - sum(n_potential_nodes_skip) / sum(n_potential_nodes)}")