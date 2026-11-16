"""Verify the exact base image/context before building the 50.2 training overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def build(context, base_image, tag):
    if tag == base_image:
        raise ValueError('Keep the old base tag intact; use a new final image tag')
    context = Path(context).resolve()
    receipt = json.loads((context / 'CONTEXT_MANIFEST.json').read_text())
    expected_files = set(receipt['files']) | {'CONTEXT_MANIFEST.json'}
    actual_files = {p.relative_to(context).as_posix() for p in context.rglob('*') if p.is_file()}
    if actual_files != expected_files:
        raise ValueError('Unexpected/missing file in context; stage a fresh context')
    for name, expected in receipt['files'].items():
        p = context / name
        if p.is_symlink() or not p.resolve().is_relative_to(context):
            raise ValueError('Unsafe context path')
        with p.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Context changed: ' + name)
    # Bind the staged Dockerfile/pins to the code being invoked, not just its receipt.
    for name in ('Dockerfile', 'audit_base.py', 'BASE_IMAGE_FILES.json', 'README.md', '.dockerignore'):
        if (context / name).read_bytes() != (HERE / name).read_bytes():
            raise ValueError('Context differs from current packaging source: ' + name)
    pin = json.loads((HERE / 'BASE_IMAGE_FILES.json').read_text())
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', base_image], text=True))[0]
    if info['Id'] not in pin['accepted_image_ids'] or info['Architecture'] != 'amd64' or info['Os'] != 'linux':
        raise ValueError('Load the audited original 50.2 archive before building this overlay')
    subprocess.run(['docker', 'build', '--platform', 'linux/amd64', '--network', 'none',
                    '--build-arg', 'BASE_IMAGE=' + base_image, '-t', tag, str(context)], check=True)
    print('BUILT_50_2_CANDIDATE; run CLI/notebook checks, export and calculate NEW archive MD5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--base-image', default='zaic-team:rtx4060-test1-fix2')
    parser.add_argument('--tag', default='zaic-team:50-2-final-training-v1')
    a = parser.parse_args()
    build(a.context, a.base_image, a.tag)
