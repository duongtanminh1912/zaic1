"""Check that the overlay preserves the actual audited inference filesystem."""
import argparse
import hashlib
import json
from pathlib import Path


def check(root=Path('/')):
    files = json.loads(Path(__file__).with_name('BASE_IMAGE_FILES.json').read_text())['files']
    for name, expected in files.items():
        p = root / name
        if p.stat().st_size != expected['bytes']:
            raise ValueError('Base image file size changed: ' + name)
        with p.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected['sha256']:
            raise ValueError('Base image file changed: ' + name)
    print('INHERITED_50_2_INFERENCE_VERIFIED', len(files), 'files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/'))
    check(parser.parse_args().root)
