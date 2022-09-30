import os
import queue
import pathlib
import argparse

import pygraphviz as pgv
import ujson as json


def visualize(model_path, model_type: str, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    base_path = pathlib.Path(model_path).with_suffix('')
    if output_dir is not None:
        base_path = os.path.join(output_dir, base_path.stem)

    with open(model_path, 'r') as f:
        js = json.load(f)
    print("Loaded.")

    if model_type == 'gbdt':
        Gs = json_to_dot_gbdt(js)
    elif model_type == 'deltaboost':
        Gs = json_to_dot_deltaboost(js)
    elif model_type == 'deltaboostv2':
        Gs = json_to_dot_deltaboostv2(js)
    else:
        assert False
    for i, G in enumerate(Gs):
        G.draw(f"{base_path}_tree{i}.svg", prog='dot')


def json_to_dot_gbdt(model):
    Gs = []
    trees = model['gbdt']['trees']
    for tree_wrapper in trees:
        G = pgv.AGraph(directed=True, ordering='in', ranksep=2)
        tree = tree_wrapper[0]
        nodes = tree['nodes']

        visiting_nodes = [0]
        next_visiting_nodes = []
        while len(visiting_nodes) > 0 or len(next_visiting_nodes) > 0:
            next_visiting_nodes = []
            for node_id in visiting_nodes:
                # insert prior node and edge to graph
                node = nodes[node_id]

                if node['is_leaf']:
                    G.add_node(f"{node_id}", shape='oval',
                               label=f"<ID={node_id}, N={node['n_instances']}<BR/>Weight={node['base_weight']:.4f}>")
                else:
                    lch_id = node['lch_index']
                    rch_id = node['rch_index']
                    next_visiting_nodes.append(lch_id)
                    next_visiting_nodes.append(rch_id)
                    G.add_node(f"{node_id}", shape='box',
                               label=f"<ID={node_id}, N={node['n_instances']}, Bid={node['split_bid']}, Fid={node['split_feature_id']}<BR/>Gain={node['gain']:.4f}>")

                parent_index = nodes[node_id]['parent_index']
                if parent_index >= 0:     # not root level
                    is_right = (nodes[parent_index]['rch_index'] == node_id)
                    color = 'red' if nodes[parent_index]['default_right'] is is_right else 'black'
                    G.add_edge(f"{parent_index}", f"{node_id}", color=color)

            visiting_nodes = next_visiting_nodes[:]
        Gs.append(G)
    return Gs


def json_to_dot_deltaboost(model):
    Gs = []
    trees = model['deltaboost']['trees']
    for tree_wrapper in trees:
        G = pgv.AGraph(directed=True, ordering='in', ranksep=2)
        tree = tree_wrapper[0]
        nodes = tree['nodes']

        visiting_nodes = [0]
        next_visiting_nodes = []
        while len(visiting_nodes) > 0 or len(next_visiting_nodes) > 0:
            # expand visiting nodes to all potential nodes
            visiting_nodes_include_potential = []
            for node_id in visiting_nodes:
                # insert prior node and edge to graph
                node = nodes[node_id]
                potential_indices = node['potential_nodes_indices']
                visiting_nodes_include_potential += potential_indices

                prior_node_id = potential_indices[0]
                last_potential_id = prior_node_id
                is_leaf = nodes[prior_node_id]['is_leaf']
                if is_leaf:
                    G.add_node(f"{prior_node_id}", shape='oval',
                               label=f"<ID={prior_node_id}, N={node['n_instances']}<BR/>Weight={node['base_weight']:.4f}>")
                else:
                    G.add_node(f"{prior_node_id}", shape='box',
                               label=f"<ID={prior_node_id}, N={node['n_instances']}, Bid={node['split_bid']}, Fid={node['split_feature_id']}<BR/>Gain={node['gain']['gain_value']:.4f}>")

                parent_index = nodes[prior_node_id]['parent_index']
                if parent_index >= 0:     # not root level
                    is_right = (nodes[parent_index]['rch_index'] == node_id)
                    color = 'red' if nodes[parent_index]['default_right'] is is_right else 'black'
                    G.add_edge(f"{parent_index}", f"{prior_node_id}", color=color)

                # add other potential nodes and edges to graph (if any)
                if len(potential_indices) > 1:
                    for potential_id in potential_indices[1:]:
                        # add main edges from the parents of potential nodes to potential nodes
                        is_leaf = nodes[potential_id]['is_leaf']
                        potential_node = nodes[potential_id]
                        if is_leaf:
                            G.add_node(f"{potential_id}", shape='oval',
                                       label=f"<ID={potential_id}, N={potential_node['n_instances']}<BR/>Weight={potential_node['base_weight']:.4f}>")
                        else:
                            G.add_node(f"{potential_id}", shape='box',
                                       label=f"<ID={potential_id}, N={potential_node['n_instances']}, Bid={potential_node['split_bid']}, Fid={potential_node['split_feature_id']}<BR/>Gain={potential_node['gain']['gain_value']:.4f}>")
                        parent_index = nodes[potential_id]['parent_index']
                        G.add_edge(f"{parent_index}", f"{potential_id}", style='invis')

                        # add short edges between potential nodes
                        G.add_edge(f"{last_potential_id}", f"{potential_id}", style='dotted', arrowsize=0.001,
                                   weight=1e2)
                        last_potential_id = potential_id

                    G.add_subgraph([f"{pid}" for pid in potential_indices],
                                   rank='same')  # same rank for potential nodes

            next_visiting_nodes = []
            for node_id in visiting_nodes_include_potential:
                node = nodes[node_id]
                if not node['is_leaf']:
                    # insert prior nodes and edges to graph, red edge means default edge
                    lch_id = node['lch_index']
                    rch_id = node['rch_index']
                    next_visiting_nodes.append(lch_id)
                    next_visiting_nodes.append(rch_id)
            visiting_nodes = next_visiting_nodes[:]
        Gs.append(G)
    return Gs


def json_to_dot_deltaboostv2(model):
    Gs = []
    trees = model['deltaboost']['trees']
    for tree_wrapper in trees:
        G = pgv.AGraph(directed=True, ordering='in', ranksep=2)
        tree = tree_wrapper[0]
        nodes = tree['nodes']

        visiting_nodes = [0]
        next_visiting_nodes = []
        while len(visiting_nodes) > 0 or len(next_visiting_nodes) > 0:
            next_visiting_nodes = []
            for node_id in visiting_nodes:
                # insert prior node and edge to graph
                node = nodes[node_id]

                if node['is_leaf']:
                    G.add_node(f"{node_id}", shape='oval',
                               label=f"<ID={node_id}, N={node['n_instances']}<BR/>Weight={node['base_weight']:.4f}>")
                else:
                    lch_id = node['lch_index']
                    rch_id = node['rch_index']
                    split_bids = node['split_nbr']['split_bids']
                    next_visiting_nodes.append(lch_id)
                    next_visiting_nodes.append(rch_id)
                    G.add_node(f"{node_id}", shape='box',
                               label=f"<ID={node_id}, N={node['n_instances']}, Bid={node['split_bid']}, "
                                     f"Fid={node['split_feature_id']}<BR/>Gain={node['gain']['gain_value']:.4f}"
                                     f"<BR/>Split_nbr=[{split_bids[0]}, {split_bids[-1]}]>")

                parent_index = nodes[node_id]['parent_index']
                if parent_index >= 0:     # not root level
                    is_right = (nodes[parent_index]['rch_index'] == node_id)
                    color = 'red' if nodes[parent_index]['default_right'] is is_right else 'black'
                    G.add_edge(f"{parent_index}", f"{node_id}", color=color)

            visiting_nodes = next_visiting_nodes[:]
        Gs.append(G)
    return Gs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--model-type', '-m', type=str, default='deltaboost')
    parser.add_argument('--output-dir', '-o', type=str, default='../fig/')
    args = parser.parse_args()
    dataset = 'cadata'
    n_trees = 10
    ratio = '1e-03'
    # visualize(f"../cache/{dataset}_tree{n_trees}_original_{ratio}_deleted.json", 'deltaboostv2', f'fig/tree_structure/{dataset}_tree{n_trees}/')
    # visualize(f"../cache/{dataset}_tree{n_trees}_original_{ratio}_deltaboost.json", 'deltaboostv2', f'fig/tree_structure/{dataset}_tree{n_trees}/')
    visualize(f"../cache/{dataset}_tree{n_trees}_retrain_{ratio}_deltaboost.json", 'deltaboostv2', f'fig/tree_structure/{dataset}_tree{n_trees}/')

    # visualize(args.path, model_type=args.model_type, output_dir=args.output_dir)

