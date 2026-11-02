from pathlib import Path
import csv
import json
import random
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from .data import load_mgtab, fingerprint, make_split, fit_standardizer, relational_undirected
from .features import extract, FEATURE_NAMES
from .trees import build_trees, validate_tree
from .models import SeBotMLS, infonce


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_config(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def signature(config):
    keys = ['seed', 'split', 'tree_depth', 'tree_neighbor_cap', 'motif_backend', 'pagerank_alpha', 'topology_backend', 'frequency_source']
    return {k: config[k] for k in keys}


def prepare(config, force=False):
    data = load_mgtab(config['data_dir'], config['expected_nodes'], config['expected_input_dim'], config['frequency_source'])
    if config['frequency_source'] == 'edge_occurrences':
        print('Frequency counts unavailable: using observed edge occurrences (documented reproduction limitation).', flush=True)
    digest = fingerprint(config['data_dir'])
    cache = Path(config['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / 'prepared.pt'
    metadata = {'format': 2, 'fingerprint': digest, 'settings': signature(config)}
    if target.exists() and not force:
        saved = torch.load(target, map_location='cpu', weights_only=True)
        if saved['metadata'] != metadata:
            raise ValueError('Cache settings/data differ. Use prepare --force or a distinct cache_dir.')
        print(f'Using validated cache: {target}')
        return target
    # Stage identity protects resumption from different inputs/settings.
    import hashlib
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    stages = cache / ('stages-' + identity)
    stages.mkdir(exist_ok=True)
    if force:
        # Overwrite/recompute only explicitly named derived stage files.
        for name in ['motifs.npy', 'global.npy', 'local-progress.npy', 'global-tree.pt', 'subgraph-progress.pt']:
            (stages / name).unlink(missing_ok=True)
    split = make_split(data.num_nodes, config['split'], config['seed'])
    seed_everything(config['seed'])
    if config['topology_backend'] == 'igraph':
        import igraph
        igraph.set_random_number_generator(random)
    raw, quantiles = extract(data, config, stages)
    structural, mean, scale = fit_standardizer(raw, split['train_idx'])
    seed_everything(config['seed'])
    trees = build_trees(data, config, stages)
    # Persist only tensors and Python primitive containers, no PyG Data/pickle objects.
    for t in trees['subgraphs']:
        # Raw upstream tree_data carries unused numpy scalar tags/degrees.
        t.pop('node_features', None)
        t.pop('node_degrees', None)
        t['label'] = 0
    payload = {'metadata': metadata, 'split': split, 'raw_structural': raw, 'structural': structural,
               'mean': mean, 'scale': scale, 'quantiles': quantiles, 'trees': trees}
    save_checkpoint(payload, target)
    (cache / 'manifest.json').write_text(json.dumps({**metadata, 'feature_names': FEATURE_NAMES, 'quantiles': quantiles,
        'shapes': {'user': list(data.x.shape), 'structural': list(raw.shape)},
        'split_sizes': {k: len(v) for k, v in split.items()}}, indent=2), encoding='utf-8')
    return target


def load_prepared(config, device):
    target = Path(config['cache_dir']) / 'prepared.pt'
    if not target.exists():
        raise FileNotFoundError('Run prepare first.')
    payload = torch.load(target, map_location='cpu', weights_only=True)
    if payload['metadata']['fingerprint'] != fingerprint(config['data_dir']) or payload['metadata']['settings'] != signature(config):
        raise ValueError('Dataset/config does not match prepared cache.')
    data = load_mgtab(config['data_dir'], config['expected_nodes'], config['expected_input_dim'], config['frequency_source'])
    data.edge_index, data.edge_type = relational_undirected(data.edge_index, data.edge_type)
    # SeBot encoders ignore PMI weights; dyadic extraction used separate counts.
    del data.edge_weight
    del data.edge_frequency
    for key, ids in payload['split'].items():
        setattr(data, key, ids)
    validate_tree(payload['trees']['global'], data.num_nodes, config['tree_depth'])
    if len(payload['trees']['subgraphs']) != data.num_nodes:
        raise ValueError('One subgraph tree per user required.')
    return data.to(device), payload['structural'].to(device), payload['trees'], payload


def get_device(name):
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Run doctor or pass --device cpu.')
    return torch.device(name)


def metrics(y, logits):
    probs = logits.softmax(-1)[:, 1].detach().cpu().numpy()
    pred = logits.argmax(-1).detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    result = {'accuracy': float(accuracy_score(y, pred)),
              'f1_bot': float(f1_score(y, pred, zero_division=0)),
              'recall_bot': float(recall_score(y, pred, zero_division=0)),
              'precision_bot': float(precision_score(y, pred, zero_division=0)),
              'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
              'roc_auc': float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
              'confusion_matrix_human_bot': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    return result


def save_checkpoint(payload, path):
    path = Path(path)
    temp = path.with_suffix('.tmp.pt')
    torch.save(payload, temp)
    temp.replace(path)


def train(config, resume=False):
    seed_everything(config['seed'])
    device = get_device(config['device'])
    data, structural, trees, payload = load_prepared(config, device)
    model = SeBotMLS(data.x.shape[1], config).to(device)
    optimizer_cls = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[config['optimizer']]
    optimizer = optimizer_cls(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    best, wait = float('inf'), 0
    start_epoch = 1
    if resume:
        latest = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
        if latest['metadata'] != payload['metadata']:
            raise ValueError('Resume checkpoint belongs to different prepared data.')
        for key in ['heads', 'dropout', 'classifier_hidden', 'tree_depth', 'edge_drop', 'lr', 'optimizer', 'weight_decay', 'lambda_ncl', 'lambda_scl', 'temperature']:
            if latest['config'][key] != config[key]:
                raise ValueError(f'Resume setting changed: {key}. Use a new output directory.')
        model.load_state_dict(latest['model'])
        optimizer.load_state_dict(latest['optimizer'])
        torch.set_rng_state(latest['rng_cpu'])
        if device.type == 'cuda' and latest['rng_cuda']:
            torch.cuda.set_rng_state_all(latest['rng_cuda'])
        best, wait, start_epoch = latest['best_val_loss'], latest['wait'], latest['epoch'] + 1
        if start_epoch > config['epochs']:
            print('Checkpoint already reached requested epoch limit.')
            return output / 'best.pt'
    started = time.time()
    with (output / 'history.csv').open('a' if resume else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'classification', 'ncl', 'scl', 'val_loss', 'val_accuracy', 'val_f1_bot'])
        if not resume:
            writer.writeheader()
        for epoch in range(start_epoch, config['epochs'] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, (u, g, c), _ = model(data, structural, trees)
            cls = torch.nn.functional.cross_entropy(logits[data.train_idx], data.y[data.train_idx])
            ncl = infonce(model.user.proj_u(u), model.user.proj_u(c), config['temperature'], config['contrastive_chunk'])
            scl = infonce(model.user.proj_g(g), model.user.proj_g(c), config['temperature'], config['contrastive_chunk'])
            loss = cls + config['lambda_ncl'] * ncl + config['lambda_scl'] * scl
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            optimizer.step()
            model.eval()
            # Upstream FACN samples noise even in eval. Fixed RNG makes each
            # validation/checkpoint evaluation reproducible without perturbing training.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                torch.manual_seed(config['seed'] + 100000)
                with torch.no_grad():
                    val_logits = model(data, structural, trees)[0][data.val_idx]
                    val_loss = torch.nn.functional.cross_entropy(val_logits, data.y[data.val_idx]).item()
                    val_metrics = metrics(data.y[data.val_idx], val_logits)
            row = dict(epoch=epoch, loss=loss.item(), classification=cls.item(), ncl=ncl.item(), scl=scl.item(),
                       val_loss=val_loss, val_accuracy=val_metrics['accuracy'], val_f1_bot=val_metrics['f1_bot'])
            writer.writerow(row)
            f.flush()
            print(f'Epoch {epoch}: loss={loss.item():.4f}, validation loss={val_loss:.4f}, bot F1={val_metrics["f1_bot"]:.4f}', flush=True)
            if val_loss < best:
                best, wait = val_loss, 0
                save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                            'val_loss': best, 'config': config, 'metadata': payload['metadata'],
                            'split': payload['split'], 'mean': payload['mean'], 'scale': payload['scale'],
                            'input_dim': data.x.shape[1]}, output / 'best.pt')
            else:
                wait += 1
            save_checkpoint({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                             'config': config, 'metadata': payload['metadata'], 'best_val_loss': best, 'wait': wait,
                             'rng_cpu': torch.get_rng_state(),
                             'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, output / 'latest.pt')
            if wait >= config['patience']:
                break
    run = {'seconds': time.time() - started, 'epochs': epoch, 'device': str(device),
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
           'peak_cuda_memory_mb': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
           'note': 'Synthetic data results are software validation only.' if config['expected_nodes'] is None else 'Reimplementation; unresolved paper choices documented in SPECIFICATION.md.'}
    (output / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    return output / 'best.pt'


def evaluate(config, checkpoint_path=None):
    device = get_device(config['device'])
    path = Path(checkpoint_path or Path(config['output_dir']) / 'best.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    # Use saved architectural settings, allowing only execution paths/device overrides.
    saved_config = ckpt['config'].copy()
    for key in ['device', 'data_dir', 'cache_dir', 'output_dir']:
        saved_config[key] = config[key]
    data, structural, trees, payload = load_prepared(saved_config, device)
    if ckpt['metadata'] != payload['metadata']:
        raise ValueError('Checkpoint belongs to a different data/split/cache.')
    model = SeBotMLS(ckpt['input_dim'], saved_config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    seed_everything(saved_config['seed'] + 100000)
    with torch.no_grad():
        logits, _, weights = model(data, structural, trees)
    result = metrics(data.y[data.test_idx], logits[data.test_idx])
    result.update(epoch=ckpt['epoch'], validation_loss=ckpt['val_loss'], test_count=len(data.test_idx),
                  dataset_fingerprint=payload['metadata']['fingerprint'], synthetic=config['expected_nodes'] is None)
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'true_label', 'predicted_label', 'bot_probability', 'dyadic_weight', 'polyadic_weight', 'local_weight', 'global_weight'])
        probs = logits.softmax(-1)[:, 1]
        for i in data.test_idx.tolist():
            writer.writerow([i, data.y[i].item(), logits[i].argmax().item(), probs[i].item(), *weights[i].cpu().tolist()])
    print(json.dumps(result, indent=2))
    return result
