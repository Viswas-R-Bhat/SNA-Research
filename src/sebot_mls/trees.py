"""SeBot encoding trees: use upstream structural entropy optimization."""
import contextlib
import io
import random
import networkx as nx
import numpy as np
import torch
from pathlib import Path
from .features import topology_graph
from .legacy.trans_graph import pool_trans_discon
from .legacy.tree import extract_layer_data
from .legacy.tree_data import extract_tree


def validate_tree(tree, n, depth):
    if tree['node_size'][0] != n or len(tree['node_size']) <= depth:
        raise ValueError('Encoding tree size/depth mismatch.')
    key = 'interLayer_edgeMat' if 'interLayer_edgeMat' in tree else 'edges'
    for level in range(1, depth + 1):
        e = torch.as_tensor(tree[key][level], dtype=torch.long)
        if key == 'edges':
            e = e.T
        if e.ndim != 2 or e.shape[0] != 2 or e.shape[1] != tree['node_size'][level - 1]:
            raise ValueError(f'Each child must have one parent (level {level}).')
        if not torch.equal(e[1].sort().values, torch.arange(tree['node_size'][level - 1])):
            raise ValueError('Encoding tree child coverage invalid.')
        if e[0].min() < 0 or e[0].max() >= tree['node_size'][level]:
            raise ValueError('Encoding tree parent indices invalid.')


def build_graph_tree(g, depth, nodelist):
    record = {'G': g.copy(), 'label': 0, 'nodelist': nodelist, 'index': 0}
    with contextlib.redirect_stdout(io.StringIO()):
        result = pool_trans_discon((record, depth))
    return result


def build_trees(data, config, stages=None):
    depth = config['tree_depth']
    g = topology_graph(data)
    print('Building global structural entropy tree; dense CPU adjacency is required by upstream...', flush=True)
    stages = Path(stages) if stages else None
    global_path = stages / 'global-tree.pt' if stages else None
    if global_path and global_path.exists():
        global_tree = torch.load(global_path, weights_only=True)
    else:
        global_record = build_graph_tree(g, depth, list(range(data.num_nodes)))
        global_tree = extract_layer_data(global_record['tree'], g, depth)
        if global_path:
            torch.save(global_tree, global_path)
    # Preserve valid COO dimensions for edgeless quotient graphs.
    global_tree['layer_edgeMat'] = [e.reshape(2, -1) for e in global_tree['layer_edgeMat']]
    validate_tree(global_tree, data.num_nodes, depth)
    # Match upstream: cap outgoing edge entries, gather incoming one-hop ego,
    # then symmetrize the extracted subgraph. This cap belongs to SeBot only.
    rng = np.random.RandomState(config['seed'])
    python_rng = random.Random(config['seed'])
    outgoing = [[] for _ in range(data.num_nodes)]
    for a, b in data.edge_index.T.tolist():
        outgoing[a].append(b)
    capped = nx.DiGraph()
    capped.add_nodes_from(range(data.num_nodes))
    for a, neighbors in enumerate(outgoing):
        if len(neighbors) > config['tree_neighbor_cap']:
            neighbors = rng.choice(neighbors, config['tree_neighbor_cap'], replace=False).tolist()
        capped.add_edges_from((a, b) for b in neighbors if a != b)
    progress_path = stages / 'subgraph-progress.pt' if stages else None
    subgraphs = torch.load(progress_path, weights_only=True) if progress_path and progress_path.exists() else []
    def persist():
        if progress_path:
            temp = progress_path.with_suffix('.tmp.pt')
            torch.save(subgraphs, temp)
            temp.replace(progress_path)
    for i in range(data.num_nodes):
        nodes = sorted({i} | set(capped.predecessors(i)))
        ego = capped.subgraph(nodes).to_undirected()
        if ego.number_of_edges() == 0:
            count = min(config['tree_neighbor_cap'], data.num_nodes - 1)
            other = python_rng.sample([v for v in range(data.num_nodes) if v != i], count)
            nodes = [i] + other
            ego = nx.Graph()
            ego.add_nodes_from(nodes)
            ego.add_edges_from((i, v) for v in other)
        # Consume the same RNG sequence even for already persisted users.
        if i < len(subgraphs):
            continue
        mapping = {v: k for k, v in enumerate(nodes)}
        ego = nx.relabel_nodes(ego, mapping)
        record = build_graph_tree(ego, depth, nodes)
        tree = extract_tree(record, depth)
        tree.pop('node_features', None)
        tree.pop('node_degrees', None)
        tree['graph_mats'] = [e.reshape(2, -1) for e in tree['graph_mats']]
        validate_tree(tree, len(nodes), depth)
        subgraphs.append(tree)
        if i % 100 == 0:
            persist()
            print(f'SeBot subgraph trees: {i}/{data.num_nodes}', flush=True)
    persist()
    return {'global': global_tree, 'subgraphs': subgraphs}
