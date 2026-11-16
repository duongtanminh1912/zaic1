"""Stage a code-only overlay with actual BTC-accessible private training URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def prepare(destination, checkpoints_url, shared_training_url):
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Preserve existing context; choose a NEW destination')
    if destination.is_relative_to(REPO) or REPO.is_relative_to(destination):
        raise ValueError('Keep the context and its private URLs outside the checkout')
    urls = {'checkpoints': checkpoints_url, 'shared_training': shared_training_url}
    for url in urls.values():
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Use an HTTPS download URL accessible to BTC')
    packages = ('baseline_50_2_training_repro', 'baseline_50_9_training_repro')
    for package in packages:
        subprocess.run([sys.executable, '-B', str(REPO / 'llm' / package / 'audit_public.py')], check=True)
    pins = json.loads((REPO / 'llm/baseline_50_2_training_repro/ARCHIVES.json').read_text())
    destination.mkdir(parents=True)
    for package in packages:
        shutil.copytree(REPO / 'llm' / package, destination / 'training_code' / package,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    access = {role: dict(pins[role], url=url) for role, url in urls.items()}
    (destination / 'training_code/DATA_ACCESS.json').write_text(json.dumps(access, indent=2) + '\n')
    shutil.copy2(HERE / 'TRAINING_README.md', destination / 'training_code/README.md')
    for name in ('Dockerfile', '.dockerignore', 'README.md', 'audit_base.py', 'BASE_IMAGE_FILES.json', 'build_submission.py'):
        shutil.copy2(HERE / name, destination / name)
    manifest = {}
    for path in destination.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                manifest[path.relative_to(destination).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (destination / 'CONTEXT_MANIFEST.json').write_text(json.dumps({'files': manifest,
        'gpu_validated': False, 'full_retraining_validated': False, 'official_submission_sent': False}, indent=2) + '\n')
    print('PREPARED_50_2_OVERLAY', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--checkpoints-url', required=True)
    parser.add_argument('--shared-training-url', required=True)
    args = parser.parse_args()
    prepare(args.destination, args.checkpoints_url, args.shared_training_url)
