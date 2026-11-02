"""Build an importable Kaggle notebook containing the modular source snapshot."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
from make_notebooks import markdown, code

ROOT = Path(__file__).resolve().parents[1]


def main():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(ROOT.rglob('*')):
            rel = p.relative_to(ROOT)
            if not p.is_file() or any(x in {'.git', '__pycache__', '.venv'} or x.endswith('.egg-info') for x in rel.parts):
                continue
            if rel.parts[0] not in {'src', 'configs', 'tests', 'docs', 'vendor'} and str(rel) not in {'pyproject.toml', 'requirements.txt', 'README.md', 'NOTICE'}:
                continue
            if p.suffix not in {'.py', '.yaml', '.toml', '.txt', '.md', '.cpp', '.h'} and p.name not in {'LICENSE', 'COPYING'}:
                continue
            if p.name == 'paper-extracted.txt':
                continue
            archive.writestr(str(rel).replace('\\', '/'), p.read_bytes())
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode()
    digest = hashlib.sha256(payload).hexdigest()
    cells = [markdown('''# SeBot-MLS — MGTAB baseline
This notebook contains the modular project source; no separate code input is needed.
The RP1-traced implementation has documented unresolved choices: it is not the authors' released code.
Read `docs/SPECIFICATION.md` and `docs/PHASE2_LEDGER.md` in the extracted project before reporting reproduction results.

## Choose the run stage
Start with `smoke` to verify the software on synthetic data. Synthetic scores are **not MGTAB results**.
Then use `prepare` once for the full MGTAB features and entropy trees, save that version's outputs,
and attach those outputs as an input to a new `benchmark` or `train` run.
`benchmark` uses ten epochs in a separate output folder; `train` retains the baseline's 200-epoch maximum.
These notebook controls and the isolated environment are engineering additions recorded in the ledger.

Enable Internet for package/data downloads. Select **GPU T4 x2** when available for training (only GPU 0 is used).
Use **Save Version → Save & Run All** for an unattended batch run. Keep the notebook private.
'''), code('''# Edit only these workflow controls; model settings remain in configs/mgtab.yaml.
STAGE = 'smoke'  # 'smoke', 'prepare', 'benchmark', 'train', or 'evaluate'
RESTORE_FROM = None  # e.g. '/kaggle/input/<saved-notebook-output>/SNA-SeBot-MLS'
assert STAGE in {'smoke', 'prepare', 'benchmark', 'train', 'evaluate'}
'''), markdown('## Extract the project\nThe compressed source snapshot below contains Python modules, configs, tests, ORCA source and licenses. It contains no credentials, datasets, PDF or local training results. You can edit the extracted modules or replace this snapshot with a newer project version.'), code(f'''from pathlib import Path
import base64, hashlib, io, os, zipfile, shutil, subprocess, sys, json, time
SOURCE_SHA256 = '{digest}'
SOURCE_B64 = '{encoded}'
source_bytes = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256
PROJECT = Path('/kaggle/working/sebot-mls-baseline')
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
    archive.extractall(PROJECT)
PERSIST = Path('/kaggle/working/SNA-SeBot-MLS')
PERSIST.mkdir(parents=True, exist_ok=True)
if RESTORE_FROM:
    restored = Path(RESTORE_FROM)
    assert restored.is_dir(), 'Attach the saved output and set its exact path'
    shutil.copytree(restored, PERSIST, dirs_exist_ok=True)
os.chdir(PROJECT)
print('Source snapshot:', SOURCE_SHA256)
print('Project:', PROJECT, '\\nOutputs:', PERSIST)
print((PROJECT / 'docs/SPECIFICATION.md').read_text())
'''), markdown('## Create the pinned environment\nAll model commands run in a separate Python environment, avoiding changes to Kaggle\'s notebook kernel. Downloads can take several minutes. The source project remains portable to Windows.'), code('''subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.6.17'], check=True)
ENV = Path('/kaggle/temp/sebot-venv')
if not (ENV / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.12', str(ENV)], check=True)
PYTHON = str(ENV / 'bin/python')
def uv_install(*args):
    subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON, *args], check=True)
uv_install('torch==2.5.1', '--index-url', 'https://download.pytorch.org/whl/cu121')
uv_install('-r', 'requirements.txt')
uv_install('--no-deps', '-e', '.')
(PROJECT / 'tools').mkdir(exist_ok=True)
subprocess.run(['g++', '-O3', '-std=c++11', 'vendor/orca/orca.cpp', '-o', 'tools/orca'], check=True)
subprocess.run([PYTHON, '-m', 'sebot_mls.cli', 'doctor'], check=True)
'''), markdown('## Configure and execute\nFull runs use all 10,199 MGTAB users and the saved 80/10/10 split. Preprocessing is CPU-intensive and can outlast a session; completed stages are reusable. Checkpoints and derived files are written under `/kaggle/working/SNA-SeBot-MLS`. Restore them using the control above after saving a version.'), code('''# Configure inside the isolated environment (the notebook kernel imports no ML libraries).
configure = r"""
import sys, json, yaml, torch
from pathlib import Path
stage, project, persist = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
cfg = yaml.safe_load((project / 'configs' / ('smoke.yaml' if stage == 'smoke' else 'mgtab.yaml')).read_text())
name = 'synthetic' if stage == 'smoke' else 'MGTAB'
cfg.update(data_dir=str(persist / 'data' / name),
           cache_dir=str(persist / ('smoke-processed' if stage == 'smoke' else 'processed')),
           output_dir=str(persist / ('smoke' if stage == 'smoke' else 'benchmark' if stage == 'benchmark' else 'outputs')),
           device='cuda' if torch.cuda.is_available() else 'cpu')
if stage in {'benchmark', 'train'} and cfg['device'] != 'cuda':
    raise RuntimeError('Select a GPU accelerator before MGTAB training.')
if stage != 'smoke':
    cfg['orca_executable'] = str(project / 'tools/orca')
if stage == 'benchmark':
    cfg['epochs'] = 10
dest = project / 'configs/kaggle-runtime.yaml'
dest.write_text(yaml.safe_dump(cfg))
print(dest.read_text())
"""
subprocess.run([PYTHON, '-c', configure, STAGE, str(PROJECT), str(PERSIST)], check=True)
CONFIG = PROJECT / 'configs/kaggle-runtime.yaml'
def run(command, *args):
    subprocess.run([PYTHON, '-u', '-m', 'sebot_mls.cli', command, '--config', str(CONFIG), *args], check=True)
if STAGE == 'smoke':
    if not (PERSIST / 'data/synthetic/features.pt').exists():
        run('synthetic')
    subprocess.run([PYTHON, '-m', 'pytest', '-q'], check=True)
    run('prepare')
    run('train', *(['--resume'] if (PERSIST / 'smoke/latest.pt').exists() else []))
    run('evaluate')
elif STAGE == 'prepare':
    if not (PERSIST / 'data/MGTAB/features.pt').exists():
        run('download')
    started = time.perf_counter()
    run('prepare')
    (PERSIST / 'preprocessing-time.json').write_text(json.dumps({'seconds': time.perf_counter()-started}))
elif STAGE in {'benchmark', 'train'}:
    assert (PERSIST / 'processed/prepared.pt').exists(), 'Run prepare, save outputs, then restore them first'
    output = PERSIST / ('benchmark' if STAGE == 'benchmark' else 'outputs')
    run('train', *(['--resume'] if (output / 'latest.pt').exists() else []))
    if STAGE == 'train':
        run('evaluate')
    report = json.loads((output / 'run.json').read_text())
    print('Training report:', report)
    if STAGE == 'benchmark':
        print('Approximate 200-epoch duration (includes startup):', report['seconds']/report['epochs']*200/60, 'minutes')
else:
    run('evaluate')
'''), markdown('## Save results for the next run\nSave the notebook version to preserve outputs. After `prepare`, use **Add Input → Notebook Outputs** to attach this saved version in a subsequent run, and set `RESTORE_FROM` to its `SNA-SeBot-MLS` folder. Do not rely on an unsaved interactive session to retain the cache. Download model outputs for the Windows demonstration; full source can be downloaded from the project folder.'), code('''folder = PERSIST / ('smoke' if STAGE == 'smoke' else 'benchmark' if STAGE == 'benchmark' else 'outputs')
if folder.exists():
    destination = shutil.make_archive(str(PERSIST / (STAGE + '-results')), 'zip', folder)
    print('Download:', destination)
    metrics = folder / 'test_metrics.json'
    if metrics.exists():
        print(('SYNTHETIC SOFTWARE CHECK — NOT MGTAB' if STAGE == 'smoke' else 'MGTAB'), metrics.read_text())
print('Workflow stage completed:', STAGE)
print('Save this version to preserve:', PERSIST)
''')]
    for i, cell in enumerate(cells):
        cell['id'] = f'sebot-kaggle-{i:02}'
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{i}', 'exec')
    nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=5)
    out = ROOT / 'notebooks/kaggle_standalone.ipynb'
    out.write_text(json.dumps(nb, indent=2), encoding='utf-8')
    print(out, out.stat().st_size, 'bytes; source snapshot', digest)


if __name__ == '__main__':
    main()
