"""CPU integration checks for serial placement, deadlines and authoritative output."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('large_predict', HERE / 'predict.py')
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


class SerialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = []
        self.calls = []
        self.env = SimpleNamespace(get_tool_calls=lambda: list(self.calls), get_state_hash=lambda: 'sdk-hash')
        def session(*args, **kwargs):
            self.events.append('session')
            return self.env
        def transcribe(paths, deadline):
            self.events.append(('asr', paths))
            return {p: 'transcript' for p in paths}, {}
        self.p = predict.Predictor.__new__(predict.Predictor)
        self.p.root = Path('/data'); self.p.split = 'train'
        self.p.result = Path(self.temp.name); self.p.assets = Path('/models')
        self.p.run_label = 'test'; self.p.ordinal = 0
        self.p.executable = Path('/runtime/llama-server')
        self.p.spec = {'qwen_file': 'qwen.gguf'}; self.p.tools = {}
        self.p.sdk = SimpleNamespace(ToolEnv=session)
        self.p.asr = SimpleNamespace(
            model=SimpleNamespace(to=lambda device: self.events.append(device)),
            torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            transcribe=transcribe)
        owner = self
        class Server:
            def __init__(self, *args, deadline):
                self.deadline = deadline
                owner.events.append('server-start')
            def __enter__(self): return self
            def __exit__(self, *args): owner.events.append('server-stop')
        self.p.server_factory = Server
        def actor(view, tools, env, backend, **kwargs):
            self.events.append('actor')
            self.assertIs(env, self.env)
            self.calls.append({'name': 'fixture.tool', 'arguments': {}})
            return {'status': 'finished', 'repairs': 0}
        self.p.actor = SimpleNamespace(infer=actor)
        self.row = {'id': 'fixture', 'latest_audio': 'audio/current.wav', 'context': {},
                    'history': [{'role': 'user', 'audio': 'audio/history.wav'}]}

    def test_serial_order_and_full_current_history_coverage(self):
        record, diagnostics = self.p.infer(self.row)
        self.assertEqual(self.events, ['session', 'cuda',
            ('asr', ['audio/current.wav', 'audio/history.wav']), 'cpu',
            'server-start', 'actor', 'server-stop', 'cpu'])
        self.assertEqual(record['tool_calls'], self.calls)
        self.assertEqual(record['state_hash'], 'sdk-hash')
        self.assertFalse(diagnostics['failure'])

    def test_asr_failure_keeps_one_session_and_never_starts_llm(self):
        self.p.asr.transcribe = lambda *args: (_ for _ in ()).throw(RuntimeError('fixture failure'))
        record, diagnostics = self.p.infer(self.row)
        self.assertEqual(self.events, ['session', 'cuda', 'cpu'])
        self.assertEqual(record['state_hash'], 'sdk-hash')
        self.assertTrue(diagnostics['failure'])
        self.assertEqual(diagnostics['error_type'], 'RuntimeError')
        self.assertTrue((self.p.result / 'test_errors.log').is_file())

    def test_expired_input_budget_does_not_load_models(self):
        with patch.object(predict.time, 'perf_counter', return_value=100.):
            record, diagnostics = self.p.infer(self.row, started=60.)
        self.assertEqual(self.events, ['session', 'cpu'])
        self.assertEqual(diagnostics['status'], 'timeout')
        self.assertEqual(record['time'], 40000)

    def test_actor_failure_keeps_real_partial_calls_and_closes_server(self):
        def fail(*args, **kwargs):
            self.calls.append({'name': 'fixture.tool', 'arguments': {}})
            raise RuntimeError('fixture actor failure')
        self.p.actor.infer = fail
        record, diagnostics = self.p.infer(self.row)
        self.assertEqual(record['tool_calls'], self.calls)
        self.assertTrue(diagnostics['failure'])
        self.assertEqual(self.events[-2:], ['server-stop', 'cpu'])

class ContextTests(unittest.TestCase):
    def test_context_has_release_manifest_and_lf_scripts(self):
        spec = importlib.util.spec_from_file_location('large_context', HERE / 'prepare_context.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            repo, assets, runtime, training = [root / n for n in ('repo', 'assets', 'runtime', 'training')]
            release = repo / 'llm/baseline_50_9_verified'
            (release / 'baseline50').mkdir(parents=True)
            (repo / 'ops/src').mkdir(parents=True)
            (release / 'baseline50/asr.py').write_bytes(b'# frozen\r\n')
            (assets / 'models/asr').mkdir(parents=True)
            weights = assets / 'models/asr/model.safetensors'
            weights.write_bytes(b'fixture-weights')
            pinned = {'files': {'models/asr/model.safetensors':
                       {'bytes': weights.stat().st_size, 'sha256': module.sha(weights)}}}
            (release / 'assets.manifest.json').write_text(json.dumps(pinned))
            code = {p.relative_to(release).as_posix(): module.sha(p) for p in release.rglob('*') if p.is_file()}
            (release / 'CODE_MANIFEST.json').write_text(json.dumps({'files_sha256': code}))
            runtime.mkdir(); server = runtime / 'llama-server'; server.write_bytes(b'\x7fELFfixture')
            (runtime / 'runtime_manifest.json').write_text(json.dumps({
                'release': 'b11800', 'platform': 'linux-amd64-cuda12.8', 'server': 'llama-server',
                'archive_sha256': 'e5d489b995c2a0965ee746a246c934be9f4ffa00c78183b2d4f644f43ae5d7a7',
                'files': {'llama-server': {'bytes': server.stat().st_size, 'sha256': module.sha(server)}}}))
            training.mkdir(); (training / 'README.md').write_text('fixture')
            with patch.object(module, 'REPO', repo):
                module.prepare(assets, runtime, training, root / 'context')
                target = root / 'context/llm/baseline_50_9_verified'
                self.assertEqual((target / 'assets.manifest.json').read_bytes(),
                                 (release / 'assets.manifest.json').read_bytes())
                self.assertEqual((target / 'baseline50/asr.py').read_bytes(), b'# frozen\r\n')
                self.assertNotIn(b'\r', (root / 'context/predict.sh').read_bytes())
                (release / 'baseline50/asr.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Release source'):
                    module.prepare(assets, runtime, training, root / 'bad-context')
                self.assertFalse((root / 'bad-context').exists())


if __name__ == '__main__':
    unittest.main()
