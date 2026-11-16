"""Reproduce Medium V3 + Qwen V2 using two explicitly pinned private inputs."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SHARED = HERE.with_name('baseline_50_9_training_repro')
sys.path.insert(0, str(SHARED))
spec = importlib.util.spec_from_file_location('shared_v2_reproduction', SHARED / 'reproduce.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
read, sha, save = shared.read, shared.sha, shared.save


def source_check():
    for name, expected in read(HERE / 'CODE_MANIFEST.json')['files'].items():
        if sha(HERE / name) != expected:
            raise ValueError('50.2 source changed: ' + name)
    if sha(SHARED / 'CODE_MANIFEST.json') != read(HERE / 'PROVENANCE.json')['shared_code_manifest_sha256']:
        raise ValueError('Wrong shared V2 source package')
    shared.source_check()


def configuration(path):
    c = shared.config(path)
    c['checkpoint_dir'] = Path(c['checkpoint_dir']).expanduser().resolve()
    if (c['work_dir'].is_relative_to(c['private_dir']) or
            c['private_dir'].is_relative_to(c['work_dir']) or
            c['work_dir'].is_relative_to(c['checkpoint_dir']) or
            c['checkpoint_dir'].is_relative_to(c['work_dir'])):
        raise ValueError('Use separate, non-overlapping private/checkpoint/work directories')
    return c


def verify(c, full=False):
    source_check()
    shared.private_check(c, full=full)
    root = c['checkpoint_dir']
    expected = read(HERE / 'EXPECTED_CHECKPOINTS.json')
    if sha(root / 'ARCHIVE_MANIFEST.json') != expected['manifest_sha256']:
        raise ValueError('Wrong 50.2 checkpoint manifest')
    for name, info in expected['files'].items():
        if not full and not name.endswith(('.json', '.py', '.safetensors')):
            continue
        p = root / name
        if p.stat().st_size != info['bytes'] or sha(p) != info['sha256']:
            raise ValueError('50.2 checkpoint input changed: ' + name)
    raw = c['private_dir']
    # Prove that reusing the shared raw/next-action package is appropriate for 50.2.
    for old, new in (
        ('training_data_indexes/train_asr_clips.json', 'prepared/train_asr_clips.json'),
        ('training_data_indexes/val_asr_clips.json', 'prepared/val_asr_clips.json'),
        ('training_data_indexes/qwen_data_manifest.json', 'prepared/DATA_MANIFEST.json'),
        ('qwen_v2_step489/adapter_model.safetensors', 'adapters/qwen/adapter_model.safetensors'),
    ):
        if sha(root / old) != sha(raw / new):
            raise ValueError('Shared training payload differs from 50.2: ' + old)
    if read(root / 'training_data_indexes/split-v1.json') != read(raw / 'prepared/split.json'):
        raise ValueError('Shared TRAIN/VAL groups differ from 50.2')
    for role, step in (('asr_medium_v3_step250', 250), ('qwen_v2_step489', 489)):
        state = read(root / role / 'trainer_state.json')
        if (state['global_step'], state['max_steps']) != (step, step):
            raise ValueError('Incomplete historical checkpoint: ' + role)
    recipe = read(HERE / 'training_recipe.json')
    if read(root / 'asr_source_evidence/config.json')['asr'] != recipe['asr']:
        raise ValueError('Medium training recipe differs from historical source')


def snapshot(c, role):
    root = Path(c[role + '_base_dir']).expanduser().resolve()
    for name, info in read(HERE / 'BASE_MODELS.json')[role]['files'].items():
        p = root / name
        if p.stat().st_size != info['bytes'] or sha(p) != info['sha256']:
            raise ValueError('Base snapshot mismatch: ' + role + '/' + name)
    return root


def medium_module(c):
    verify(c)
    base = snapshot(c, 'asr')
    for name, version in read(SHARED / 'ENVIRONMENT.json')['required_training_packages'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('Historical training requires ' + name + '==' + version)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,garbage_collection_threshold:0.8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    import medium_training as m
    m.HERE = c['work_dir'] / 'asr_medium'
    m.HERE.mkdir(exist_ok=True)
    m.BASE_DIR = base
    m.RAW_TRAIN = c['private_dir'] / 'raw'

    def guard(data, full=False):
        verify(c)
        if data.resolve() != c['private_dir'] / 'prepared':
            raise ValueError('Use the verified prepared dataset')
        return {'config': {'asr': read(HERE / 'training_recipe.json')['asr']}}

    m.verify = guard
    return m


def merge_medium(c, adapter=None):
    verify(c)
    base = snapshot(c, 'asr')
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    adapter = Path(adapter).resolve() if adapter else c['checkpoint_dir'] / 'asr_medium_v3_step250'
    dest = c['work_dir'] / 'merged/asr'
    if dest.exists():
        raise FileExistsError('Preserve existing merged weights; choose a new work directory')
    model = WhisperForConditionalGeneration.from_pretrained(
        base, local_files_only=True, use_safetensors=False, dtype=torch.float16,
        device_map={'': 'cpu'})
    model = PeftModel.from_pretrained(model, adapter, device_map={'': 'cpu'}).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(dest, safe_serialization=True, max_shard_size='2GB')
    # The archived compatibility note explicitly uses the original pinned Medium processor.
    WhisperProcessor.from_pretrained(base, local_files_only=True).save_pretrained(dest)
    save(dest / 'MERGE_PROVENANCE.json', {
        'role': 'asr', 'base_revision': read(HERE / 'BASE_MODELS.json')['asr']['revision'],
        'adapter_sha256': sha(adapter / 'adapter_model.safetensors'),
        'parameters': sum(p.numel() for p in model.parameters()), 'dtype': 'float16',
    })


def unpack(archive, dest, role):
    pin = read(HERE / 'ARCHIVES.json')[role]
    archive, dest = Path(archive).resolve(), Path(dest).resolve()
    if archive.stat().st_size != pin['bytes'] or sha(archive) != pin['sha256']:
        raise ValueError('Wrong archive for ' + role)
    import zipfile
    if dest.exists():
        raise FileExistsError('Choose a NEW destination')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Duplicate archive member')
        for info in z.infolist():
            n = info.filename
            if (not (dest / n).resolve().is_relative_to(dest) or Path(n).is_absolute()
                    or ':' in n or '\\' in n or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe archive member')
        z.extractall(dest)
    print('UNPACKED', role, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('unpack', 'verify', 'prepare', 'retrieval', 'profile',
                                        'train-qwen', 'train-asr', 'merge-qwen', 'merge-asr', 'export'))
    parser.add_argument('--config')
    parser.add_argument('--archive')
    parser.add_argument('--dest')
    parser.add_argument('--role', choices=('checkpoints', 'shared_training'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--adapter-dir')
    a = parser.parse_args()
    if a.stage == 'unpack':
        if not all((a.archive, a.dest, a.role)):
            parser.error('unpack requires --archive, --dest, --role')
        unpack(a.archive, a.dest, a.role)
        return
    if not a.config:
        parser.error('--config is required')
    c = configuration(a.config)
    verify(c, full=a.stage == 'verify')
    if a.stage == 'verify':
        print('VERIFIED: Medium250, Qwen489, shared input parity, complete private payloads')
    elif a.stage == 'prepare':
        shared.prepare(c)
    elif a.stage == 'retrieval':
        p = c['work_dir'] / 'prepared_rebuilt/retrieval_index.json'
        if sha(p) != sha(c['private_dir'] / 'prepared/retrieval_index.json'):
            raise ValueError('Run prepare first; retrieval must match the TRAIN-fit bank')
        print('RETRIEVAL_VERIFIED', p)
    elif a.stage == 'profile':
        shared.profile(c)
    elif a.stage == 'train-qwen':
        shared.train(c, 'qwen', a.smoke, a.resume)
    elif a.stage == 'train-asr':
        medium_module(c).training(c['private_dir'] / 'prepared', c['work_dir'] / 'asr_medium/outputs', a.smoke, a.resume)
    elif a.stage == 'merge-qwen':
        shared.merge(c, 'qwen', a.adapter_dir)
    elif a.stage == 'merge-asr':
        merge_medium(c, a.adapter_dir)
    elif a.stage == 'export':
        shared.export(c)


if __name__ == '__main__':
    main()
