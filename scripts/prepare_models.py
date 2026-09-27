#!/usr/bin/env python
"""Download pretrained checkpoints and external model sources from their providers."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

# Fixed upstream versions used by the adapters.
HF_FILES = {
    'cbramod': ('weighting666/CBraMod', '500543c7e30bda1b22bfd51a49301b238dee21fd', 'pretrained_weights.pth'),
    'luna': ('PulpBio/LUNA', '999c1af0fd6fbfb43ed47169c61fe9faa49cfe9f', 'LUNA_base.safetensors'),
    'codebrain': ('YjMajy/CodeBrain', 'bef08d2fdb1759685371cc635aad21ce59163689', 'CodeBrain.pth'),
}
URL_FILES = {
    'biot': ('https://raw.githubusercontent.com/ycq091044/BIOT/d138e32634e52ae9fa6ec98ac9c4087b14ca869a/pretrained-models/EEG-six-datasets-18-channels.ckpt', 'EEG-six-datasets-18-channels.ckpt'),
    'labram': ('https://raw.githubusercontent.com/935963004/LaBraM/5f5ec3e702199ef0f16ee0bbaa8c2997cb77b786/checkpoints/labram-base.pth', 'labram-base.pth'),
}
SOURCES = {
    'codebrain': ('https://github.com/jingyingma01/CodeBrain.git', '22d350caf68246d2fda4f630ef837420db3fb130', 'CodeBrain'),
    'csbrain': ('https://github.com/yuchen2199/CSBrain.git', '185aee55b24d0410a830df8dd08d03f675616998', 'CSBrain'),
}
REVE = {
    'brain-bzh/reve-base': 'fa9a2163a4b7c0a42c8e28b56077ef9c368944dc',
    'brain-bzh/reve-positions': 'befa5b57a455b77cf302daf610c2e9ed8140bace',
}
MODELS = ('biot', 'labram', 'cbramod', 'eegpt', 'luna', 'reve', 'codebrain', 'csbrain')


def download_url(url, destination):
    if destination.is_file() and destination.stat().st_size:
        return
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    temporary = destination.with_suffix(destination.suffix + '.partial')
    with session.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise RuntimeError(f'Expected a checkpoint, received HTML from {url}')
        with temporary.open('wb') as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        length = response.headers.get('Content-Length')
        if length and temporary.stat().st_size != int(length):
            raise RuntimeError(f'Incomplete download: {destination.name}; rerun this command.')
    temporary.replace(destination)


def checkout(url, revision, destination):
    if (destination / '.git').is_dir():
        current = subprocess.run(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True, capture_output=True)
        if current.returncode == 0 and current.stdout.strip() != revision:
            raise RuntimeError(f'{destination} is at {current.stdout.strip()}, expected {revision}. Use another output directory to keep your checkout unchanged.')
        if current.returncode == 0:
            return
    elif destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f'{destination} already contains files; use an empty model directory.')
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / '.git').is_dir():
        subprocess.run(['git', 'init', '-q', str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'remote', 'add', 'origin', url], check=True)
    subprocess.run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)


def prepare(model, root):
    from huggingface_hub import hf_hub_download, snapshot_download
    if model in SOURCES:
        url, revision, name = SOURCES[model]
        checkout(url, revision, root / 'sources' / name)
    if model in HF_FILES:
        repo, revision, filename = HF_FILES[model]
        if not (root / filename).is_file():
            hf_hub_download(repo, filename, revision=revision, local_dir=root)
    elif model in URL_FILES:
        url, filename = URL_FILES[model]
        download_url(url, root / filename)
    elif model == 'eegpt':
        for filename in ('config.json', 'model.safetensors'):
            if not (root / 'eegpt' / filename).is_file():
                hf_hub_download('braindecode/eegpt-pretrained', filename,
                                revision='e41cb3ae2ce4fd9eb736862292c91f8128d15618', local_dir=root / 'eegpt')
    elif model == 'reve':
        for repo, revision in REVE.items():
            snapshot_download(repo, revision=revision, cache_dir=root / 'hf_cache' / 'hub',
                              allow_patterns=['*.json', '*.py', '*.safetensors'])
    elif model == 'csbrain':
        destination = root / 'CSBrain.pth'
        if not destination.is_file():
            import gdown
            temporary = destination.with_suffix('.pth.partial')
            result = gdown.download(id='1Z67je1HQhClyG9XER_zJzFMAKSVNmnF0', output=str(temporary), quiet=False)
            if result is None or not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError('CSBrain download failed. Retry later, or obtain CSBrain.pth from the provider linked in docs/MODELS.md and place it in ' + str(root))
            temporary.replace(destination)


def write_environment(root):
    variables = {
        'IESSEEG_PRETRAINED_DIR': root,
        'IESSEEG_EEGPT_UPSTREAM': root / 'eegpt',
        'IESSEEG_HF_CACHE': root / 'hf_cache',
        'HF_HOME': root / 'hf_cache',
        'IESSEEG_CODEBRAIN_ROOT': root / 'sources' / 'CodeBrain',
        'IESSEEG_CSBRAIN_ROOT': root / 'sources' / 'CSBrain',
    }
    (root / 'env.sh').write_text('# Model locations generated by scripts/prepare_models.py\n' +
                               ''.join(f'export {key}={shlex.quote(str(value))}\n' for key, value in variables.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('pretrained'))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    for model in args.models:
        print(f'Preparing {model}...', flush=True)
        try:
            prepare(model, root)
        except Exception as error:
            failures.append(model)
            print(f'{model}: {error}', file=sys.stderr, flush=True)
    if failures:
        parser.exit(1, 'Models still needed: ' + ', '.join(failures) + '. Rerun the same command to retry.\n')
    write_environment(root)
    print(f'Models ready. Run: source {shlex.quote(str(root / "env.sh"))}')


if __name__ == '__main__':
    main()
