import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import torch
from .engine import read_config, prepare, train, evaluate


def download(folder):
    import gdown
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    archive = target.parent / 'MGTAB.zip'
    if not archive.exists():
        result = gdown.download(id='1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe', output=str(archive), quiet=False, resume=True)
        if not result:
            raise RuntimeError('Official Drive download failed; use the README browser link.')
    if not zipfile.is_zipfile(archive):
        raise ValueError('Download is not a ZIP archive; remove the incomplete archive and download manually.')
    with zipfile.ZipFile(archive) as z:
        for name in ['features.pt', 'labels_bot.pt', 'edge_index.pt', 'edge_type.pt', 'edge_weight.pt']:
            candidates = [m for m in z.infolist() if Path(m.filename).name == name]
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one {name} in archive.')
            dest = target / name
            if dest.exists():
                print(f'Preserving existing {dest}')
                continue
            with z.open(candidates[0]) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    print(f'Official MGTAB tensors placed in {target}')


def synthetic(folder):
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError('Synthetic target must be empty, to protect existing data.')
    rng = np.random.RandomState(123)
    n = 40
    pairs = [(i, (i+1) % n) for i in range(n)]
    pairs += [(i, (i+2) % n) for i in range(0, n, 3)]
    pairs += [(b, a) for a, b in pairs]
    pairs += [(0, 0)]
    values = {'features.pt': torch.tensor(rng.normal(size=(n, 24)), dtype=torch.float32),
              'labels_bot.pt': torch.arange(n) % 2,
              'edge_index.pt': torch.tensor(pairs).T,
              'edge_type.pt': torch.tensor(rng.randint(0, 7, len(pairs))),
              'edge_weight.pt': torch.tensor(rng.randint(1, 5, len(pairs)), dtype=torch.float32)}
    for name, value in values.items():
        torch.save(value, target / name)
    print('Created synthetic software fixture; it is not MGTAB.')


def doctor():
    import sys
    import torch_geometric
    result = {'python': sys.version, 'torch': torch.__version__, 'pyg': torch_geometric.__version__,
              'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda}
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name(0)
        result['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 2**30
        x = torch.randn(16, 16, device='cuda', requires_grad=True)
        x.square().mean().backward()
        result['cuda_backward_pass'] = bool(torch.isfinite(x.grad).all())
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description='SeBot-MLS MGTAB baseline')
    parser.add_argument('command', choices=['doctor', 'download', 'synthetic', 'prepare', 'train', 'evaluate'])
    parser.add_argument('--config', default='configs/mgtab.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--force', action='store_true', help='Rebuild derived cache only')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume', action='store_true', help='Continue latest.pt, preserving optimizer and RNG state')
    args = parser.parse_args()
    if args.command == 'doctor':
        return doctor()
    config = read_config(args.config)
    if args.device:
        config['device'] = args.device
    if args.command == 'download':
        download(config['data_dir'])
    elif args.command == 'synthetic':
        synthetic(config['data_dir'])
    elif args.command == 'prepare':
        print(prepare(config, args.force))
    elif args.command == 'train':
        print(train(config, args.resume))
    elif args.command == 'evaluate':
        evaluate(config, args.checkpoint)


if __name__ == '__main__':
    main()
