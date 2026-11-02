"""RP1 equations 7-15. No labels are used by feature extraction."""
from itertools import combinations
from pathlib import Path
import subprocess
import tempfile
import networkx as nx
import numpy as np
import torch

from .data import INTIMACY_WEIGHTS

FEATURE_NAMES = [
    'intimacy_0_25', 'intimacy_25_50', 'intimacy_50_75', 'intimacy_75_95', 'intimacy_95_100', 'intimacy_total',
    'reciprocity_unidirectional', 'reciprocity_0_10', 'reciprocity_10_90', 'reciprocity_90_100', 'reciprocity_complete', 'reciprocity_self_loop',
    'motif_path3', 'motif_triangle3', 'motif_star4', 'motif_path4', 'motif_paw4', 'motif_cycle4', 'motif_diamond4', 'motif_clique4',
    'ego_nodes', 'ego_edges', 'ego_density', 'constraint', 'average_clustering', 'transitivity',
    'pagerank', 'eigenvector', 'betweenness', 'closeness']


def topology_graph(data):
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(data.edge_index.T.tolist())
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def dyadic(data):
    n = data.num_nodes
    intimacy, directed = {}, {}
    self_loops = np.zeros(n)
    frequencies = data.edge_frequency if 'edge_frequency' in data else torch.ones(data.edge_index.shape[1])
    for (a, b), r, frequency in zip(data.edge_index.T.tolist(), data.edge_type.tolist(), frequencies.tolist()):
        # Interaction frequency is separate from released PMI edge weights.
        value = INTIMACY_WEIGHTS[r] * frequency
        if value <= 0:
            continue
        key = (a, b)
        intimacy[key] = intimacy.get(key, 0.0) + value
        if r < 5:
            directed[key] = directed.get(key, 0.0) + value
            if a == b:
                self_loops[a] += 1
    q = np.quantile(list(intimacy.values()), [.25, .5, .75, .95]) if intimacy else np.zeros(4)
    out = np.zeros((n, 12), dtype=np.float64)
    for (a, b), value in intimacy.items():
        out[a, np.searchsorted(q, value, side='left')] += 1
        out[a, 5] += 1
    pair_keys = {(a, b) for a, b in directed} | {(b, a) for a, b in directed}
    reciprocity = {}
    for a, b in pair_keys:
        if a != b:
            ab, ba = directed.get((a, b), 0), directed.get((b, a), 0)
            reciprocity[a, b] = 1 - 2 * abs(ab / (ab + ba) - .5)
    interior = [v for v in reciprocity.values() if 0 < v < 1]
    rq = np.quantile(interior, [.1, .9]) if interior else np.zeros(2)
    for (a, b), value in reciprocity.items():
        col = 6 if value == 0 else 10 if np.isclose(value, 1, atol=1e-12, rtol=0) else 7 + np.searchsorted(rq, value, side='left')
        out[a, col] += 1
    out[:, 11] = self_loops
    return out, {'intimacy_quantiles': q.tolist(), 'reciprocity_interior_quantiles': rq.tolist()}


def graphlet_column(g):
    degree = tuple(sorted(dict(g.degree()).values()))
    return {
        (1, 1, 2): 0, (2, 2, 2): 1, (1, 1, 1, 3): 2, (1, 1, 2, 2): 3,
        (1, 2, 2, 3): 4, (2, 2, 2, 2): 5, (2, 2, 3, 3): 6, (3, 3, 3, 3): 7,
    }.get(degree)


def motif_counts(g, backend='enumerate', executable=None):
    n = len(g)
    if backend == 'orca':
        if not executable or not Path(executable).is_file():
            raise FileNotFoundError('Set orca_executable to a compiled ORCA executable (see README).')
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'graph.in', Path(tmp) / 'orbits.out'
            src.write_text(f'{n} {g.number_of_edges()}\n' + ''.join(f'{a} {b}\n' for a, b in g.edges()), encoding='ascii')
            subprocess.run([str(Path(executable).resolve()), 'node', '4', str(src), str(dst)], check=True)
            orbits = np.loadtxt(dst, ndmin=2)
            if orbits.shape != (n, 15):
                raise ValueError(f'Invalid ORCA output shape {orbits.shape}')
            return np.column_stack([orbits[:, ids].sum(1) for ids in [[1, 2], [3], [6, 7], [4, 5], [9, 10, 11], [8], [12, 13], [14]]])
    if backend != 'enumerate':
        raise ValueError('motif_backend must be enumerate or orca.')
    result = np.zeros((n, 8), dtype=np.float64)
    adj = {i: set(g[i]) for i in g}
    # Enumerate connected induced subsets with their unique minimum vertex.
    # Deduplication is per minimum vertex, never across the whole graph.
    for root in g:
        states = {frozenset([root])}
        for size in range(2, 5):
            new = set()
            for subset in states:
                frontier = set.union(*(adj[v] for v in subset)) - subset
                for v in frontier:
                    if v > root:
                        new.add(subset | {v})
            states = new
            if size >= 3:
                for subset in states:
                    col = graphlet_column(g.subgraph(subset))
                    if col is not None:
                        result[list(subset), col] += 1
        if root % 100 == 0:
            print(f'Exact motifs: {root}/{n}', flush=True)
    return result


def local_features(g, backend='networkx', progress_path=None):
    progress_path = Path(progress_path) if progress_path else None
    saved = np.load(progress_path) if progress_path and progress_path.exists() else np.full((len(g), 6), np.nan)
    if saved.shape != (len(g), 6):
        raise ValueError('Invalid saved local-feature shape.')
    def persist(out):
        if progress_path:
            temp = progress_path.with_suffix('.tmp.npy')
            np.save(temp, out)
            temp.replace(progress_path)
    if backend == 'igraph':
        import igraph as ig
        graph = ig.Graph(n=len(g), edges=list(g.edges()), directed=False)
        graph.vs['original_id'] = list(range(len(g)))
        out = saved
        for i in range(len(g)):
            if np.isfinite(out[i]).all():
                continue
            ego = graph.induced_subgraph(graph.neighborhood(i, order=2))
            center = ego.vs.find(original_id=i).index
            c = ego.constraint(vertices=[center])[0]
            out[i] = [ego.vcount(), ego.ecount(), ego.density(), c if np.isfinite(c) else 0,
                      ego.transitivity_avglocal_undirected(mode='zero'), ego.transitivity_undirected(mode='zero')]
            if i % 100 == 0:
                persist(out)
                print(f'Exact two-hop ego features (C backend): {i}/{len(g)}', flush=True)
        # An isolated one-vertex ego has undefined density in igraph.
        out[~np.isfinite(out)] = 0
        persist(out)
        return out
    if backend != 'networkx':
        raise ValueError('topology_backend must be networkx or igraph.')
    out = saved
    for i in g:
        if np.isfinite(out[i]).all():
            continue
        ego = nx.ego_graph(g, i, radius=2)
        c = nx.constraint(ego, nodes=[i])[i]
        # Burt constraint is undefined for isolated users; use zero, recorded in spec.
        out[i] = [len(ego), ego.number_of_edges(), nx.density(ego), c if np.isfinite(c) else 0,
                  nx.average_clustering(ego), nx.transitivity(ego)]
        if i % 100 == 0:
            persist(out)
            print(f'Exact two-hop ego features: {i}/{len(g)}', flush=True)
    persist(out)
    return out


def global_features(g, alpha=.85, backend='networkx'):
    if backend == 'igraph':
        import igraph as ig
        graph = ig.Graph(n=len(g), edges=list(g.edges()), directed=False)
        pr = np.asarray(graph.pagerank(damping=alpha, directed=False))
        if graph.ecount():
            ev = np.asarray(graph.eigenvector_centrality(directed=False, scale=False))
            ev /= np.linalg.norm(ev)
        else:
            ev = np.zeros(len(g))
        bc = np.asarray(graph.betweenness(directed=False))
        cc = np.asarray(graph.closeness(normalized=True))
        cc[~np.isfinite(cc)] = 0
        return np.column_stack([pr, ev, bc, cc])
    if backend != 'networkx':
        raise ValueError('topology_backend must be networkx or igraph.')
    pr = nx.pagerank(g, alpha=alpha, max_iter=1000, tol=1e-10)
    if g.number_of_edges() == 0:
        ev = dict.fromkeys(g, 0.0)
    else:
        ev = nx.eigenvector_centrality(g, max_iter=5000, tol=1e-10)
    bc = nx.betweenness_centrality(g, normalized=False)  # all sources, no sampling
    cc = nx.closeness_centrality(g, wf_improved=False)
    return np.array([[pr[i], ev[i], bc[i], cc[i]] for i in g])


def extract(data, config, stages=None):
    g = topology_graph(data)
    d, quantiles = dyadic(data)
    stages = Path(stages) if stages else None
    def cached(name, compute):
        path = stages / f'{name}.npy' if stages else None
        if path and path.exists():
            print(f'Resuming completed {name} stage.', flush=True)
            return np.load(path)
        value = compute()
        if path:
            temp = path.with_suffix('.tmp.npy')
            np.save(temp, value)
            temp.replace(path)
        return value
    p = cached('motifs', lambda: motif_counts(g, config['motif_backend'], config.get('orca_executable')))
    l = local_features(g, config['topology_backend'], stages / 'local-progress.npy' if stages else None)
    print('Exact global centralities (may take substantial time)...', flush=True)
    glob = cached('global', lambda: global_features(g, config['pagerank_alpha'], config['topology_backend']))
    raw = np.concatenate([d, p, l, glob], 1)
    if raw.shape != (data.num_nodes, 30) or not np.isfinite(raw).all():
        raise ValueError('Expected finite [N,30] structural features.')
    return torch.tensor(raw, dtype=torch.float32), quantiles
