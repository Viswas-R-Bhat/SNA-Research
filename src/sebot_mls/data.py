from pathlib import Path
import hashlib
import numpy as np
import torch
from torch_geometric.data import Data

RELATIONS = ['follower', 'friend', 'mention', 'reply', 'quote', 'url', 'hashtag']
INTIMACY_WEIGHTS = np.array([0.60, 0.80, 1.00, 0.80, 0.60, 0.35, 0.30])


def fingerprint(folder):
    h = hashlib.sha256()
    for name in ['features.pt', 'labels_bot.pt', 'edge_index.pt', 'edge_type.pt', 'edge_weight.pt']:
        h.update(name.encode())
        with (Path(folder) / name).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
    frequency_path = Path(folder) / 'interaction_frequency.pt'
    if frequency_path.exists():
        h.update(b'interaction_frequency.pt')
        with frequency_path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def load_mgtab(folder, expected_nodes=None, expected_input_dim=None, frequency_source='edge_occurrences'):
    folder = Path(folder)
    required = ['features.pt', 'labels_bot.pt', 'edge_index.pt', 'edge_type.pt', 'edge_weight.pt']
    missing = [p for p in required if not (folder / p).exists()]
    if missing:
        raise FileNotFoundError(f'Missing {missing} in {folder}. Run download or place official tensors here.')
    # Official distribution consists of tensors; never execute an arbitrary serialized object.
    def read(name):
        return torch.as_tensor(torch.load(folder / name, map_location='cpu', weights_only=True))
    x, y = read('features.pt').float(), read('labels_bot.pt').long().flatten()
    edges = read('edge_index.pt').long()
    types, weights = read('edge_type.pt').long().flatten(), read('edge_weight.pt').double().flatten()
    n = len(y)
    if x.ndim != 2 or x.shape[0] != n or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('Expected x[N,D], labels[N], edges[2,E]; MGTAB-large is not supported.')
    if len(types) != edges.shape[1] or len(weights) != len(types):
        raise ValueError('Edge types/weights do not align with edge_index.')
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('Edge endpoints out of range.')
    if not torch.isfinite(x).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Nonfinite features or invalid interaction weights.')
    if not set(y.tolist()) <= {0, 1} or not set(types.tolist()) <= set(range(7)):
        raise ValueError('Expected bot labels 0/1 and relation IDs 0..6.')
    if expected_nodes is not None and n != expected_nodes:
        raise ValueError(f'Expected {expected_nodes} nodes, received {n}.')
    if expected_input_dim is not None and x.shape[1] != expected_input_dim:
        raise ValueError(f'Expected {expected_input_dim} user features, received {x.shape[1]}.')
    if frequency_source == 'file':
        frequency = read('interaction_frequency.pt').double().flatten()
        if frequency.shape != weights.shape or not torch.isfinite(frequency).all() or (frequency < 0).any():
            raise ValueError('interaction_frequency.pt must be a finite nonnegative [E] tensor aligned with original edges.')
    elif frequency_source == 'edge_occurrences':
        # The standardized release does not provide explicit event frequencies.
        # Count each observed edge occurrence once; never use PMI as a count.
        frequency = torch.ones_like(weights)
    else:
        raise ValueError('frequency_source must be file or edge_occurrences.')
    return Data(x=x, y=y, edge_index=edges, edge_type=types, edge_weight=weights, edge_frequency=frequency)


def make_split(n, ratios, seed):
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError('Three positive split ratios summing to one required.')
    ids = np.random.RandomState(seed).permutation(n)
    a, b = int(ratios[0] * n), int(sum(ratios[:2]) * n)
    parts = [torch.from_numpy(v.copy()).long() for v in [ids[:a], ids[a:b], ids[b:]]]
    if any(len(v) == 0 for v in parts):
        raise ValueError('Dataset too small for nonempty splits.')
    return dict(zip(['train_idx', 'val_idx', 'test_idx'], parts))


def relational_undirected(edges, types):
    chunks = []
    relations = []
    for r in range(7):
        e = edges[:, types == r]
        e = torch.cat([e, e.flip(0)], dim=1).unique(dim=1)
        chunks.append(e)
        relations.append(torch.full((e.shape[1],), r, dtype=torch.long))
    return torch.cat(chunks, 1), torch.cat(relations)


def fit_standardizer(raw, train_idx):
    train = raw[train_idx]
    mean, scale = train.mean(0), train.std(0, unbiased=False)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (raw - mean) / scale, mean, scale
