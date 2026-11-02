"""Check full-user MLS attention/loss memory without inventing SeBot trees.

This is a component stress check on actual MGTAB topology and random inputs,
not model training or benchmark evaluation. Full SeBot memory also depends on
the as-yet-unprepared encoding trees.
"""
import json
from pathlib import Path
import time
import torch
from sebot_mls.data import load_mgtab, relational_undirected
from sebot_mls.models import HGNN, AFusion, infonce


def main():
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    data = load_mgtab('data/MGTAB', 10199, 788)
    edges, _ = relational_undirected(data.edge_index, data.edge_type)
    edges = edges.to('cuda')
    n = data.num_nodes
    model = HGNN(4, .3, query_chunk=256).cuda().train()
    fusion = AFusion(4, .3).cuda().train()
    x = torch.randn(n, 30, device='cuda', requires_grad=True)
    a = torch.randn(n, 32, device='cuda', requires_grad=True)
    b = torch.randn(n, 32, device='cuda', requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result, weights = fusion(model(x, edges))
    loss = result.square().mean() + .1 * infonce(a, b, .1, 256)
    loss.backward()
    torch.cuda.synchronize()
    report = {'kind': 'component_capacity_check_random_inputs_not_training', 'nodes': n,
              'relation_symmetrized_edges': edges.shape[1], 'seconds': time.time()-started,
              'gpu': torch.cuda.get_device_name(),
              'peak_memory_mb': torch.cuda.max_memory_allocated()/2**20,
              'loss_finite': bool(torch.isfinite(loss)),
              'gradients_finite': bool(torch.isfinite(x.grad).all() and torch.isfinite(a.grad).all()),
              'limitation': 'Does not include SeBot tree/FACN activations; full model capacity unverified.'}
    folder = Path('outputs/validation')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'gpu_capacity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
