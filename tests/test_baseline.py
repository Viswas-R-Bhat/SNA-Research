from itertools import combinations
from pathlib import Path
import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from sebot_mls.data import make_split, fit_standardizer, load_mgtab, fingerprint
from sebot_mls.features import motif_counts, graphlet_column, dyadic, local_features, global_features
from sebot_mls.models import infonce, HGNN, AFusion, LevelEncoder


def brute_counts(g):
    out = np.zeros((len(g), 8))
    for size in [3, 4]:
        for nodes in combinations(g.nodes, size):
            sub = g.subgraph(nodes)
            if nx.is_connected(sub):
                col = graphlet_column(sub)
                out[list(nodes), col] += 1
    return out


@pytest.mark.parametrize('seed', range(5))
def test_motifs_against_independent_subsets(seed):
    g = nx.gnp_random_graph(9, .4, seed=seed)
    expected = brute_counts(g)
    np.testing.assert_array_equal(motif_counts(g), expected)
    exe = Path('tools/orca.exe')
    if exe.exists():
        np.testing.assert_array_equal(motif_counts(g, 'orca', str(exe)), expected)


def test_split_and_no_scaler_leakage():
    split = make_split(10199, [.8, .1, .1], 42)
    assert [len(v) for v in split.values()] == [8159, 1020, 1020]
    ids = torch.cat(list(split.values()))
    assert ids.unique().numel() == 10199
    raw = torch.tensor([[1., 0.], [3., 0.], [10000., 1.]])
    result, mean, scale = fit_standardizer(raw, torch.tensor([0, 1]))
    torch.testing.assert_close(mean, torch.tensor([2., 0.]))
    torch.testing.assert_close(scale, torch.tensor([1., 1.]))
    torch.testing.assert_close(result[:2, 0], torch.tensor([-1., 1.]))


def test_dyadic_boundaries_and_reciprocity():
    data = Data(x=torch.zeros(4, 1), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 2]]),
                edge_type=torch.tensor([2, 2, 3, 2]), edge_weight=torch.tensor([2., 2., 1., 1.]))
    raw, meta = dyadic(data)
    assert raw.shape == (4, 12)
    assert raw[0, 10] == raw[1, 10] == 1  # complete reciprocity
    assert raw[1, 6] == raw[2, 6] == 1    # unilateral pair counted for each participant
    assert raw[2, 11] == 1              # self-loop separated
    np.testing.assert_array_equal(raw[:, :5].sum(1), raw[:, 5])
    assert np.isfinite(raw).all()


def test_centrality_and_ego_definitions():
    g = nx.path_graph(4)
    local = local_features(g)
    assert local[0, 0] == 3  # radius=2
    assert local[0, 1] == 2
    assert local[0, 5] == 0
    glob = global_features(g)
    np.testing.assert_allclose(glob[:, 2], [0, 2, 2, 0])
    assert np.isfinite(local).all() and np.isfinite(glob).all()


def test_compiled_topology_backend_matches_reference():
    g = nx.gnp_random_graph(12, .3, seed=18)
    g.add_node(12)  # isolate
    np.testing.assert_allclose(local_features(g, 'igraph'), local_features(g), atol=1e-10)
    np.testing.assert_allclose(global_features(g, backend='igraph'), global_features(g), atol=1e-7)


def test_contrastive_chunking_values_and_gradients():
    torch.manual_seed(3)
    a, b = torch.randn(6, 4, requires_grad=True), torch.randn(6, 4, requires_grad=True)
    z = torch.cat([torch.nn.functional.normalize(a, dim=1), torch.nn.functional.normalize(b, dim=1)])
    scores = z @ z.T / .1
    scores = scores.masked_fill(torch.eye(12, dtype=torch.bool), -torch.inf)
    expected = torch.nn.functional.cross_entropy(scores, (torch.arange(12) + 6) % 12)
    gradients = torch.autograd.grad(expected, [a, b], retain_graph=True)
    actual = infonce(a, b, .1, 3)
    torch.testing.assert_close(actual, expected)
    for g1, g2 in zip(torch.autograd.grad(actual, [a, b]), gradients):
        torch.testing.assert_close(g1, g2, atol=1e-5, rtol=1e-5)


def test_hgnn_fusion_shapes_and_gradients():
    raw = torch.randn(8, 30, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    levels = HGNN(4, .0)(raw, edges)
    assert levels.shape == (8, 4, 16)
    fused, weights = AFusion(4, .0)(levels)
    assert fused.shape == (8, 32)
    torch.testing.assert_close(weights.sum(1), torch.ones(8))
    fused.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_exact_cross_node_attention_query_chunks():
    torch.manual_seed(4)
    full = LevelEncoder([6, 6], 4, 0, query_chunk=100)
    chunked = LevelEncoder([6, 6], 4, 0, query_chunk=3)
    chunked.load_state_dict(full.state_dict())
    a = torch.randn(8, 12, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    result1, result2 = full(a, edges), chunked(b, edges)
    torch.testing.assert_close(result1, result2, atol=1e-6, rtol=1e-5)
    result1.square().sum().backward()
    result2.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-4)


def test_explicit_frequency_does_not_use_pmi(tmp_path):
    from sebot_mls.cli import synthetic
    synthetic(tmp_path)
    data = load_mgtab(tmp_path)
    raw, _ = dyadic(data)
    torch.save(data.edge_weight * .123, tmp_path / 'edge_weight.pt')
    changed = load_mgtab(tmp_path)
    np.testing.assert_array_equal(dyadic(changed)[0], raw)
    before = fingerprint(tmp_path)
    frequency = torch.arange(changed.edge_index.shape[1], dtype=torch.float32) + 1
    torch.save(frequency, tmp_path / 'interaction_frequency.pt')
    assert fingerprint(tmp_path) != before
    actual = load_mgtab(tmp_path, frequency_source='file')
    torch.testing.assert_close(actual.edge_frequency, frequency.double())
    assert not np.allclose(dyadic(actual)[1]['intimacy_quantiles'], dyadic(changed)[1]['intimacy_quantiles'])


def test_resume_exact_local_features(tmp_path):
    g = nx.gnp_random_graph(10, .4, seed=5)
    expected = local_features(g, 'igraph')
    partial = expected.copy()
    partial[4:] = np.nan
    path = tmp_path / 'progress.npy'
    np.save(path, partial)
    actual = local_features(g, 'igraph', path)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(np.load(path), expected)


def test_disconnected_encoding_trees():
    from sebot_mls.trees import build_trees, validate_tree
    data = Data(x=torch.zeros(12, 3), edge_index=torch.empty(2, 0, dtype=torch.long), edge_type=torch.empty(0, dtype=torch.long))
    trees = build_trees(data, {'tree_depth': 4, 'seed': 42, 'tree_neighbor_cap': 10})
    validate_tree(trees['global'], 12, 4)
    assert len(trees['subgraphs']) == 12
    for tree in trees['subgraphs']:
        validate_tree(tree, len(tree['nodelist']), 4)


def test_training_resume_matches_uninterrupted_cpu(tmp_path):
    from sebot_mls.cli import synthetic
    from sebot_mls.engine import read_config, prepare, train, evaluate
    config = read_config('configs/smoke.yaml')
    config.update(data_dir=str(tmp_path / 'data'), cache_dir=str(tmp_path / 'cache'),
                  output_dir=str(tmp_path / 'resumed'), device='cpu', epochs=1)
    synthetic(config['data_dir'])
    prepare(config)
    train(config)
    config['epochs'] = 2
    train(config, resume=True)
    resumed = torch.load(tmp_path / 'resumed/latest.pt', weights_only=True)
    config['output_dir'] = str(tmp_path / 'uninterrupted')
    train(config)
    full = torch.load(tmp_path / 'uninterrupted/latest.pt', weights_only=True)
    for name in full['model']:
        torch.testing.assert_close(full['model'][name], resumed['model'][name], rtol=1e-5, atol=1e-6)
    assert resumed['epoch'] == 2
    result = evaluate(config)
    assert result['synthetic'] and result['test_count'] == 4
