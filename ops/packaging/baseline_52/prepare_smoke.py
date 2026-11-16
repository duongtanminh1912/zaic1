"""Prepare five checksum-verified TRAIN-fit cases for the Large 52 image."""
import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'ops/src'))
from lifemate.runtime.configuration import strict_json
from lifemate.runtime.smoke_inputs import prepare_smoke, inside, digest


def prepare(dataset, assets, destination, count=5):
    dataset, assets, destination = map(lambda p: Path(p).resolve(), (dataset, assets, destination))
    if destination.exists():
        raise FileExistsError('Use a new smoke directory; preserve earlier results')
    if not 1 <= count <= 5:
        raise ValueError('Smoke is bounded to one through five TRAIN-fit cases')
    pinned = strict_json((REPO / 'llm/baseline_52_release/inference/assets.manifest.json').read_text())
    split_path = assets / 'private_data/split.json'
    info = pinned['files']['private_data/split.json']
    if split_path.stat().st_size != info['bytes'] or digest(split_path) != info['sha256']:
        raise ValueError('Wrong Large 52 split')
    ids = strict_json(split_path.read_text())['train_fit'][:count]
    if len(ids) != count:
        raise ValueError('Insufficient TRAIN-fit IDs')
    # Common validator checks official TRAIN JSONL/catalog/SDK, index/split,
    # PCM audio and FIT membership before creating anything model-facing.
    prepared = destination / 'prepared'
    report = prepare_smoke(dataset, assets / 'private_data', ids, prepared,
                           max_samples=count, assets_manifest=pinned)
    source, target = dataset / 'train', destination / 'data/train'
    target.mkdir(parents=True)
    shutil.copytree(source / 'zaic_tools', target / 'zaic_tools',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(source / 'train_tools_catalog.json', target / 'train_tools_catalog.json')
    with (target / 'train.jsonl').open('x', encoding='utf-8') as output:
        for line in (prepared / 'prepared_audio.jsonl').read_text().splitlines():
            item = strict_json(line)
            row = {'id': item['id'], 'audio': item['latest_audio'],
                   'context': item['context'], 'history': item['history']}
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            refs = [row['audio']] + [t['audio'] for t in row['history'] if t['role'] == 'user']
            refs.append('train_environment/' + row['id'] + '/state.json')
            for relative in refs:
                original = inside(source, relative)
                copied = target / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                if not copied.exists():
                    shutil.copy2(original, copied)
    # State is copied as bytes for SDK, never parsed or exposed to the actor.
    return {'samples': report['samples'], 'audio_count': len(report['audio']),
            'data_mount': str(destination / 'data'), 'model_inference_run': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.dataset, a.assets, a.destination)))
