from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import SAGEConv
from .legacy.sep_u import SEP_U
from .legacy.sep_g import SEP_G
from .legacy.backbone import BotRGCN


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU())


class LevelEncoder(nn.Module):
    """Equations 16-27. Attention is across users, not feature columns."""
    def __init__(self, parts, heads, dropout, query_chunk=256):
        super().__init__()
        self.parts = parts
        self.encoders = nn.ModuleList([mlp(size, 16 // len(parts)) for size in parts])
        self.sage = SAGEConv(16, 16)
        self.attention = nn.MultiheadAttention(16, heads, dropout=dropout, batch_first=True)
        self.output = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.query_chunk = query_chunk

    def forward(self, raw, edges):
        x = torch.cat([m(p) for m, p in zip(self.encoders, raw.split(self.parts, dim=1))], 1)
        h = F.relu(self.sage(x, edges)).unsqueeze(0)
        chunks = []
        for start in range(0, h.shape[1], self.query_chunk):
            def attend(h, start=start):
                # Every query sees all users. Only query rows are chunked.
                return self.attention(h[:, start:start+self.query_chunk], h, h, need_weights=False)[0]
            chunks.append(checkpoint(attend, h, use_reentrant=False) if self.training else attend(h))
        a = torch.cat(chunks, 1).squeeze(0)
        return self.norm(self.output(a))


class HGNN(nn.Module):
    def __init__(self, heads, dropout, query_chunk=256):
        super().__init__()
        self.levels = nn.ModuleList([LevelEncoder(p, heads, dropout, query_chunk) for p in [[6, 6], [2, 6], [3, 3], [4]]])

    def forward(self, raw, edges):
        return torch.stack([m(p, edges) for m, p in zip(self.levels, raw.split([12, 8, 6, 4], 1))], 1)


class AFusion(nn.Module):
    """Equations 28-34; four level tokens per user, each 16-dimensional."""
    def __init__(self, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(16, heads, dropout=dropout, batch_first=True)
        self.dynamic = nn.Linear(64, 4)
        self.level_importance = nn.Parameter(torch.zeros(4))
        self.output = nn.Linear(16, 32)

    def forward(self, levels):
        e = levels.flatten(1)
        attended = self.attention(levels, levels, levels, need_weights=False)[0]
        alpha = self.dynamic(e).softmax(-1)
        weights = alpha * self.level_importance.sigmoid()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        merged = (weights.unsqueeze(-1) * attended).sum(1)
        return self.output(merged), weights


class SeBotBranch(nn.Module):
    """Retain SEP_U, SEP_G and FACN from the original SeBot source."""
    def __init__(self, input_dim, config):
        super().__init__()
        args = SimpleNamespace(
            dataset='MGTAB', input_dim=input_dim, hidden_dim=32, proj_dim=16,
            des_num=768, tweet_num=768, prop_num=5, cat_num=3,
            num_classes=2, num_features=32, node_num=0, num_relations=7,
            num_blocks=2, num_convs=3, tree_depth=config['tree_depth'],
            conv='GCN', link_input=False, global_pooling='average',
            dropout=config['dropout'], conv_dropout=config['dropout'], pooling_dropout=config['dropout'],
            pe=config['edge_drop'], pf=.2, device=torch.device('cpu'),
        )
        self.args = args
        self.global_encoder = SEP_U(args)
        self.subgraph_encoder = SEP_G(args)
        self.relational_encoder = BotRGCN(args)
        self.proj_u = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 32))
        self.proj_g = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 32))
        self.subgraph_batch = config['subgraph_batch']

    def forward(self, data, trees):
        self.args.device = data.x.device
        self.relational_encoder.node_num = data.num_nodes
        u = self.global_encoder({'data': data, 'layer_data': trees['global']})
        c = self.relational_encoder(data)
        chunks = []
        for start in range(0, data.num_nodes, self.subgraph_batch):
            sub = trees['subgraphs'][start:start + self.subgraph_batch]
            # Recompute subgraph activations during backward to bound GPU memory.
            def run(x, sub=sub):
                return self.subgraph_encoder(sub, x)
            chunks.append(checkpoint(run, data.x, use_reentrant=False) if self.training else run(data.x))
        g = torch.cat(chunks, 0)
        return torch.cat([u, g, c], 1), (u, g, c)


class SeBotMLS(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        if 16 % config['heads']:
            raise ValueError('Attention heads must divide 16.')
        self.user = SeBotBranch(input_dim, config)
        self.structural = HGNN(config['heads'], config['dropout'], config['attention_query_chunk'])
        self.fusion = AFusion(config['heads'], config['dropout'])
        self.classifier = nn.Sequential(nn.Linear(128, config['classifier_hidden']), nn.LeakyReLU(), nn.Linear(config['classifier_hidden'], 2))

    def forward(self, data, structural, trees):
        user, views = self.user(data, trees)
        levels = self.structural(structural, data.edge_index)
        structural, weights = self.fusion(levels)
        z = torch.cat([user, structural], 1)
        return self.classifier(z), views, weights


def infonce(a, b, temperature=.1, chunk=256):
    """Exact upstream symmetric NT-Xent, including all 2N-2 negatives.

    Query-row chunking with checkpointing retains the same loss/gradients and
    avoids retaining a full [2N,2N] matrix on a 6GB GPU.
    """
    if temperature <= 0 or chunk <= 0 or a.shape != b.shape:
        raise ValueError('Invalid contrastive inputs.')
    n = len(a)
    z = torch.cat([F.normalize(a, dim=1), F.normalize(b, dim=1)], 0)
    total = z.new_zeros(())
    for start in range(0, 2 * n, chunk):
        end = min(start + chunk, 2 * n)
        def block(z, start=start, end=end):
            scores = z[start:end] @ z.T / temperature
            rows = torch.arange(end - start, device=z.device)
            ids = torch.arange(start, end, device=z.device)
            scores[rows, ids] = -torch.inf
            positive = (ids + n) % (2 * n)
            return (torch.logsumexp(scores, 1) - scores[rows, positive]).sum()
        total = total + checkpoint(block, z, use_reentrant=False)
    return total / (2 * n)
