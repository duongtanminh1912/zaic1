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
    def test_stages_52_release_both_training_links_and_rejects_wrong_assets(self):
        import shutil
        spec = importlib.util.spec_from_file_location('context52', HERE / 'prepare_context.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        original_repo = module.REPO
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve();repo=root/'repo'
            release=repo/'llm/baseline_52_release'
            shutil.copytree(original_repo/'llm/baseline_52_release',release,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            (repo/'ops/src').mkdir(parents=True)
            assets=root/'assets';(assets/'models/asr').mkdir(parents=True)
            first=assets/'models/asr/model-00001-of-00002.safetensors';first.write_bytes(b'shard1')
            second=first.with_name('model-00002-of-00002.safetensors');second.write_bytes(b'shard2')
            index=first.with_name('model.safetensors.index.json');index.write_text('{}')
            qwen=assets/'models/qwen/model52.gguf';qwen.parent.mkdir();qwen.write_bytes(b'dpo52')
            pinned={'files':{p.relative_to(assets).as_posix():{'bytes':p.stat().st_size,'sha256':module.sha(p)} for p in (first,second,index,qwen)}}
            manifest=release/'inference/assets.manifest.json';manifest.write_text(json.dumps(pinned))
            code_path=release/'CODE_MANIFEST.json';code=json.loads(code_path.read_bytes())
            code['files']['inference/assets.manifest.json']=module.sha(manifest);code_path.write_text(json.dumps(code))
            runtime=root/'runtime';runtime.mkdir();server=runtime/'llama-server';server.write_bytes(b'ELF')
            (runtime/'runtime_manifest.json').write_text(json.dumps({
              'release':'b11800','platform':'linux-amd64-cuda12.8','server':'llama-server',
              'archive_sha256':'e5d489b995c2a0965ee746a246c934be9f4ffa00c78183b2d4f644f43ae5d7a7',
              'files':{'llama-server':{'bytes':server.stat().st_size,'sha256':module.sha(server)}}}))
            urls={'base_training_url':'https://example.org/base.zip','dpo_training_url':'https://example.org/dpo.zip'}
            with patch.object(module,'REPO',repo):
                module.prepare(assets,runtime,release,root/'context',**urls)
                staged=root/'context'
                access=json.loads((staged/'training_code/DATA_ACCESS.json').read_text())
                self.assertEqual(access['archives']['04_TRAINING_DPO52.zip']['url'],urls['dpo_training_url'])
                self.assertEqual(len(access['archives']),2)
                self.assertTrue((staged/'training_code/baseline_52_release/training/dpo/reproduce.py').is_file())
                self.assertTrue((staged/'private_assets/models/asr'/second.name).is_file())
                self.assertNotIn(b'\r',(staged/'predict.sh').read_bytes())
                qwen.write_bytes(b'old-parent')
                with self.assertRaisesRegex(ValueError,'Model asset'):
                    module.prepare(assets,runtime,release,root/'bad-model',**urls)
                self.assertFalse((root/'bad-model').exists());qwen.write_bytes(b'dpo52')
                second.unlink()
                with self.assertRaises(FileNotFoundError):
                    module.prepare(assets,runtime,release,root/'missing-shard',**urls)
                second.write_bytes(b'shard2')
                with self.assertRaisesRegex(ValueError,'HTTPS'):
                    module.prepare(assets,runtime,release,root/'missing-dpo',base_training_url=urls['base_training_url'],dpo_training_url='')

if __name__=='__main__':unittest.main()
