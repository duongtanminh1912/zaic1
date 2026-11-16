"""Regression checks for checkpoint finalization and submission artifact guards."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(m)
    finally:
        sys.dont_write_bytecode = before
    return m


class Final50Tests(unittest.TestCase):
    def test_medium_repeated_final_save_evaluates_once(self):
        source = REPO / 'llm/baseline_50_2_training_repro/medium_training.py'
        training = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'training')
        progress = next(n for n in training.body if isinstance(n, ast.ClassDef) and n.name == 'Progress')
        callback = next(n for n in progress.body if isinstance(n, ast.FunctionDef) and n.name == 'on_save')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'checkpoints/checkpoint-250'
            checkpoint.mkdir(parents=True)
            for n in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                      'scheduler.pt', 'trainer_state.json', 'rng_state.pth'):
                (checkpoint / n).write_bytes(b'fixture')
            (root / 'val_asr_clips.json').write_text('[]')
            writes, evaluations = [], []

            def save_weights(path, **kwargs):
                path.mkdir(parents=True, exist_ok=False)
                (path / 'adapter_model.safetensors').write_bytes(b'fixture')
                writes.append(path)

            def save_json(path, value):
                path.write_text(json.dumps(value))

            env = {'handled_saves': set(), 'run': root, 'data': root, 'smoke': False,
                   'model': SimpleNamespace(save_pretrained=save_weights),
                   'processor': SimpleNamespace(save_pretrained=lambda p: p.mkdir(parents=True)),
                   'sha': lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                   'atomic': save_json, 'json_new': save_json,
                   'evaluate_asr': lambda *args: evaluations.append(args) or {'wer_percent': 10.79},
                   'read': lambda p: json.loads(p.read_text()), 'torch': None, 'np': None,
                   'val_history': []}
            exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), 'exec'), env)
            state = SimpleNamespace(global_step=250, max_steps=250, epoch=2.0)
            control = SimpleNamespace(should_save=True)
            self.assertIs(env['on_save'](None, None, state, control), control)
            self.assertIs(env['on_save'](None, None, state, control), control)
            self.assertEqual(len(evaluations), 1)
            self.assertEqual(len(env['val_history']), 1)
            self.assertEqual(len(writes), 2)  # epoch adapter and best adapter

    def test_unpack_rejects_traversal_even_with_matching_archive_hash(self):
        m = module('repro50_test', REPO / 'llm/baseline_50_2_training_repro/reproduce.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'payload.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escape.txt', 'unsafe')
            pin = {'checkpoints': {'bytes': archive.stat().st_size, 'sha256': m.sha(archive)}}
            with patch.object(m, 'read', return_value=pin):
                with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                    m.unpack(archive, root / 'new', 'checkpoints')
            self.assertFalse((root / 'escape.txt').exists())
            self.assertFalse((root / 'new').exists())

    def test_medium_snapshot_refuses_substituted_weights(self):
        m = module('repro50_pin_test', REPO / 'llm/baseline_50_2_training_repro/reproduce.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pytorch_model.bin').write_bytes(b'wrong')
            pin = {'asr': {'files': {'pytorch_model.bin': {'bytes': 5, 'sha256': '0' * 64}}}}
            with patch.object(m, 'read', return_value=pin):
                with self.assertRaisesRegex(ValueError, 'Base snapshot mismatch'):
                    m.snapshot({'asr_base_dir': str(root)}, 'asr')

    def test_builder_rejects_wrong_model_image_before_docker_build(self):
        m = module('build50_test', REPO / 'ops/packaging/baseline_50_2_final/build_submission.py')
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)
            names = ('Dockerfile', 'audit_base.py', 'BASE_IMAGE_FILES.json', 'README.md', '.dockerignore')
            for n in names:
                (context / n).write_bytes((m.HERE / n).read_bytes())
            receipt = {n: hashlib.sha256((context / n).read_bytes()).hexdigest() for n in names}
            (context / 'CONTEXT_MANIFEST.json').write_text(json.dumps({'files': receipt}))
            wrong = [{'Id': 'sha256:wrong-model', 'Architecture': 'amd64', 'Os': 'linux'}]
            with patch.object(m.subprocess, 'check_output', return_value=json.dumps(wrong)), \
                    patch.object(m.subprocess, 'run') as build:
                with self.assertRaisesRegex(ValueError, 'original 50.2 archive'):
                    m.build(context, 'old-tag', 'new-tag')
                build.assert_not_called()
            # Context changes are also rejected before any Docker operation.
            (context / 'Dockerfile').write_text('FROM wrong-model\n')
            with patch.object(m.subprocess, 'check_output') as inspect:
                with self.assertRaisesRegex(ValueError, 'Context changed'):
                    m.build(context, 'old-tag', 'new-tag')
                inspect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
