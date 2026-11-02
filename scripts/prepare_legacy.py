"""Create reviewed compatibility copies, preserving upstream sources and license.

Changes: relative imports; remove unused torch_sparse fused path (we supply COO
edges); remove training/CLI code; preserve SEP_U, SEP_G, FACN and tree algorithms.
"""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'vendor' / 'SEBot'
DST = ROOT / 'src' / 'sebot_mls' / 'legacy'


def copy(source, target, replace=None, cut=None, remove_sparse=False):
    text = (SRC / source).read_text(encoding='utf-8')
    if cut:
        text = text.split(cut)[0]
    for a, b in (replace or {}).items():
        text = text.replace(a, b)
    # Preserve rank-two empty COO edge matrices in modern Torch.
    text = text.replace('torch.LongTensor(es).T', 'torch.LongTensor(es).reshape(-1, 2).T')
    text = text.replace('torch.LongTensor(edges).T', 'torch.LongTensor(edges).reshape(-1, 2).T')
    text = text.replace('torch.LongTensor(edges).transpose(0, 1)', 'torch.LongTensor(edges).reshape(-1, 2).transpose(0, 1)')
    if remove_sparse:
        text = text.replace('from torch_sparse import SparseTensor, matmul', '')
        # Remove the unused optimized sparse dispatch entirely; PyG falls back
        # to the original message() on COO edge_index, with identical addition.
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        spans = [(n.lineno - 1, n.end_lineno) for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == 'message_and_aggregate']
        for start, end in sorted(spans, reverse=True):
            del lines[start:end]
        text = ''.join(lines)
    (DST / target).write_text('# Adapted from 846468230/SEBot, Apache-2.0. See NOTICE.\n' + text, encoding='utf-8')


def main():
    DST.mkdir(parents=True, exist_ok=True)
    (DST / '__init__.py').write_text('', encoding='utf-8')
    copy('SEPN/sep_u.py', 'sep_u.py', remove_sparse=True)
    copy('SEPG/sep_g.py', 'sep_g.py', remove_sparse=True)
    copy('BackBone/rgcn.py', 'rgcn.py', cut="if __name__ == '__main__':")
    copy('SEPN/codingTree.py', 'codingTree.py')
    copy('SEPN/tree.py', 'tree.py', {'from codingTree import PartitionTree': 'from .codingTree import PartitionTree'}, cut='def load_tree(')
    copy('SEPG/utils.py', 'tree_utils.py', cut='def get_nodelist(')
    copy('SEPG/trans_graph.py', 'trans_graph.py', {'from utils import load_graph, PartitionTree': 'from .tree_utils import PartitionTree'}, cut='def struct_tree(')
    copy('SEPG/data.py', 'tree_data.py', cut='def integrate_label(')
    copy('main_mgtab.py', 'backbone.py', {
        'from SEPN.sep_u import SEP_U': 'from .sep_u import SEP_U',
        'from SEPG.sep_g import SEP_G': 'from .sep_g import SEP_G',
        'from BackBone.rgcn import FACNConv as RGCNConv': 'from .rgcn import FACNConv as RGCNConv',
    }, cut='class SEBot(')
    (DST / 'LICENSE').write_text((SRC / 'LICENSE').read_text(), encoding='utf-8')


if __name__ == '__main__':
    main()
