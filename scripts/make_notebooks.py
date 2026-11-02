"""Build portable, parameterized notebook entry points (no hosted execution)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(True)}


def build(platform):
    cells = [markdown(f'''# SeBot-MLS on {platform}
Same Python project/model as the Windows baseline. Run preprocessing once,
persist its cache, then train. Do not present synthetic smoke scores as MGTAB results.
See docs/SPECIFICATION.md for unresolved paper choices. Kaggle is the recommended
optional cloud training environment; code is authored/tested locally in VS Code.
Select a GPU accelerator for training, and enable Internet for dependency setup.
The notebook uses one GPU, even if two T4s are allocated.
''')]
    if platform == 'Colab':
        cells += [markdown('Upload the supplied `sebot-mls-baseline.zip` when prompted. Mount Drive to preserve datasets, cache and checkpoints across runtime resets.'),
                  code('from google.colab import drive, files\ndrive.mount("/content/drive")\nuploaded = files.upload()\n'),
                  code('''from pathlib import Path
import zipfile, os
archive = Path(next(name for name in uploaded if name.endswith('.zip')))
with zipfile.ZipFile(archive) as z:
    z.extractall('/content/sebot-source')
PROJECT = Path('/content/sebot-source/sebot-mls-baseline')
assert (PROJECT / 'pyproject.toml').exists()
PERSIST = Path('/content/drive/MyDrive/SNA-SeBot-MLS')
PERSIST.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
''')]
    else:
        cells += [markdown('Attach the source ZIP as a private Kaggle input. Set `SOURCE_ZIP` to its actual path. Optional: attach the official five MGTAB tensors as another private input. `/kaggle/input` is read-only; derived output stays in `/kaggle/working`. Save a version with **Save & Run All** for a batch run.'),
                  code('''from pathlib import Path
import zipfile, os
SOURCE_ZIP = Path('/kaggle/input/YOUR-PROJECT-DATASET/sebot-mls-baseline.zip')
assert SOURCE_ZIP.exists(), 'Set SOURCE_ZIP to the uploaded archive path'
with zipfile.ZipFile(SOURCE_ZIP) as z:
    z.extractall('/kaggle/working/sebot-source')
PROJECT = Path('/kaggle/working/sebot-source/sebot-mls-baseline')
assert (PROJECT / 'pyproject.toml').exists()
PERSIST = Path('/kaggle/working/SNA-SeBot-MLS')
PERSIST.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
''')]
    cells += [markdown('Install the verified package versions. The pinned runtime requires Python 3.10-3.12. If the hosted default changes, use a matching Python environment rather than silently changing the baseline dependencies. Installation may take several minutes; GPU quotas/session resources are managed by the provider.'),
              code('''import sys, subprocess
assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Use Python 3.10-3.12'
subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.5.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', '.'], check=True)
'''),
              markdown('Restart the notebook kernel/runtime after installation if NumPy/PyTorch were already imported. Then rerun the path cell above, without repeating upload/extraction.'),
              code('''import sys, subprocess, yaml
(PROJECT / 'tools').mkdir(exist_ok=True)
subprocess.run(['g++', '-O3', '-std=c++11', 'vendor/orca/orca.cpp', '-o', 'tools/orca'], check=True)
subprocess.run([sys.executable, '-m', 'sebot_mls.cli', 'doctor'], check=True)
'''),
              code('''# Change RAW_DATA to a folder containing the five official MGTAB tensors,
# or leave it here and use the download cell below. Keep the same seed and splits.
RAW_DATA = PERSIST / 'data' / 'MGTAB'
CACHE = PERSIST / 'processed'
OUTPUT = PERSIST / 'outputs'
cfg = yaml.safe_load((PROJECT / 'configs/mgtab.yaml').read_text())
cfg.update(data_dir=str(RAW_DATA), cache_dir=str(CACHE), output_dir=str(OUTPUT),
           device='cuda', orca_executable=str(PROJECT / 'tools/orca'))
CONFIG = PROJECT / 'configs/cloud-runtime.yaml'
CONFIG.write_text(yaml.safe_dump(cfg))
print(CONFIG.read_text())
'''),
              code('''def run(command, *extra):
    subprocess.run([sys.executable, '-m', 'sebot_mls.cli', command, '--config', str(CONFIG), *extra], check=True)
if not (RAW_DATA / 'features.pt').exists():
    run('download')
'''),
              markdown('**Preprocess once.** Exact ego statistics, centralities and entropy trees can take substantial CPU time. A GPU does not accelerate all of this stage. Stop here after saving the cache if the session deadline is close, and train from the saved cache in a new session.'),
              code("run('prepare')\n"),
              markdown('Train with the paper settings; checkpoint selection uses validation loss. Evaluation is a separate command and does not influence training. On Kaggle use Save & Run All. On Colab outputs below persist in Drive.'),
              code("run('train', *(['--resume'] if (OUTPUT / 'latest.pt').exists() else []))\n"),
              code("run('evaluate')\nprint((OUTPUT / 'test_metrics.json').read_text())\n"),
              code('''# Bundle source settings and outputs for the local demonstration.
# Copy raw data and the prepared cache separately if they are not already local.
import shutil
result_zip = shutil.make_archive(str(PERSIST / 'sebot-training-outputs'), 'zip', OUTPUT)
print(result_zip)
''')]
    if platform == 'Colab':
        cells.append(code("from google.colab import files\nfiles.download(result_zip)\n"))
    else:
        cells.append(markdown('The output ZIP and cache appear under `/kaggle/working/SNA-SeBot-MLS`. Save/download them using the notebook Output panel. GPU allocation/quotas can change; inspect your current settings.'))
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    folder = ROOT / 'notebooks'
    folder.mkdir(exist_ok=True)
    for platform in ['Colab', 'Kaggle']:
        nb = build(platform)
        for i, cell in enumerate(nb['cells']):
            cell['id'] = f'{platform.lower()}-{i:02}'
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'{platform}-cell-{i}', 'exec')
        (folder / f'{platform.lower()}.ipynb').write_text(json.dumps(nb, indent=2), encoding='utf-8')
