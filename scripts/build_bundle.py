"""Package portable source without environments, datasets or generated scores."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'output' / 'sebot-mls-baseline.zip'
SKIP = {'.venv', '.git', '__pycache__', '.pytest_cache', 'data', 'outputs'}


def main():
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(p in SKIP or p.endswith('.egg-info') for p in relative.parts):
                continue
            if path.name == 'random_graphs.zip' or path.suffix == '.pyc':
                continue
            if relative.parts[0] == 'docs' and (path.name.startswith('page-') or path.name == 'paper-extracted.txt'):
                continue
            z.write(path, Path(ROOT.name) / relative)
    with zipfile.ZipFile(OUT) as z:
        assert z.testzip() is None
        assert 'sebot-mls-baseline/README.md' in z.namelist()
        assert 'sebot-mls-baseline/notebooks/kaggle.ipynb' in z.namelist()
        assert 'sebot-mls-baseline/notebooks/colab.ipynb' in z.namelist()
        assert not any('/data/' in p or '/.venv/' in p or '/.git/' in p for p in z.namelist())
        print(f'Validated {len(z.namelist())} source files')
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f'{OUT}\n{OUT.stat().st_size:,} bytes\nSHA256 {digest}')


if __name__ == '__main__':
    main()
