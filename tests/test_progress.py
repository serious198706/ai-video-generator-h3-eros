import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from h3_service.config import Settings
from h3_service.infer.engine import Engine
from h3_service.infer.manifest import sha256, verify, BASE_REVISION, DIFFUSERS_REVISION, COMPONENTS


class ProgressTests(unittest.TestCase):
    def test_hash_progress_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'weight'
            file.write_bytes(b'model contents')
            events = []
            self.assertEqual(sha256(file, progress=lambda *args: events.append(args)), hashlib.sha256(file.read_bytes()).hexdigest())
            self.assertEqual(events[0][0], 0)
            self.assertEqual(events[-1][0], file.stat().st_size)
            cancelled = threading.Event()
            cancelled.set()
            with self.assertRaises(InterruptedError):
                sha256(file, cancelled=cancelled)

    def test_verify_progress_keeps_integrity_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for component in COMPONENTS:
                (root / component).mkdir()
            index, eros = root / 'modular_model_index.json', root / 'eros.safetensors'
            index.write_text('{}')
            eros.write_bytes(b'eros')
            (root / 'h3-manifest.json').write_text(json.dumps({'revision': BASE_REVISION,
                'files': {'modular_model_index.json': sha256(index)}}))
            settings = Settings(model_dir=root, eros_file=eros)
            messages = []
            with patch('h3_service.infer.manifest.EROS_SHA256', sha256(eros)):
                verify(settings, progress=messages.append)
                self.assertTrue(any('GiB to read' in m for m in messages))
                self.assertEqual(messages[-1], 'All weight hashes verified')
                index.write_text('corrupt')
                with self.assertRaisesRegex(ValueError, 'weight hash mismatch'):
                    verify(settings, progress=messages.append)

    def test_benchmark_can_omit_only_duplicate_model_probes(self):
        versions = {'torch': '2.11.0+cu130', 'torchvision': '0.26.0+cu130',
                    'transformers': '5.8.0', 'accelerate': '1.14.0'}
        distribution = MagicMock()
        distribution.read_text.return_value = json.dumps({'vcs_info': {'commit_id': DIFFUSERS_REVISION}})
        with tempfile.TemporaryDirectory() as temp, patch('shutil.which', return_value='/bin/ffprobe'), \
                patch('importlib.metadata.distribution', return_value=distribution), \
                patch('importlib.metadata.version', side_effect=versions.__getitem__), \
                patch('h3_service.infer.engine.verify') as verify_mock:
            engine = Engine(Settings(work_dir=Path(temp)))
            with patch.object(engine, 'phase') as phase:
                engine.preflight(threading.Event(), probe_models=False)
                self.assertEqual([c.args[0] for c in phase.call_args_list], ['probe-runtime'])
                verify_mock.assert_called_once()
            with patch.object(engine, 'phase') as phase:
                engine.preflight(threading.Event())
                self.assertEqual([c.args[0] for c in phase.call_args_list],
                                 ['probe-runtime', 'probe-condition', 'probe-generate'])

    def test_benchmark_existing_directory_is_not_modified(self):
        import benchmark
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / 'prompt.txt'
            prompt.write_text('a calm landscape')
            output = root / 'previous'
            output.mkdir()
            for name in ('job.json', 'conditioning.safetensors', 'report.json'):
                (output / name).write_text('keep')
            with patch('sys.argv', ['benchmark.py', '--image', str(root / 'image.png'),
                                   '--prompt-file', str(prompt), '--output', str(output)]):
                self.assertEqual(benchmark.main(), 1)
            for name in ('job.json', 'conditioning.safetensors', 'report.json'):
                self.assertEqual((output / name).read_text(), 'keep')

    def test_benchmark_reports_request_validation_fields(self):
        import benchmark
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / 'prompt.txt'
            prompt.write_text('a calm landscape')
            output = root / 'invalid-request'
            stderr = io.StringIO()
            with patch('sys.argv', ['benchmark.py', '--image', str(root / 'image.png'),
                                   '--prompt-file', str(prompt), '--output', str(output),
                                   '--resolution', '1080p', '--duration', '15']), \
                    patch('sys.stderr', stderr):
                self.assertEqual(benchmark.main(), 2)
            self.assertFalse(output.exists())
            message = stderr.getvalue()
            self.assertIn('Request validation failed', message)
            self.assertIn('duration', message)
            self.assertIn('less than or equal to 14', message)
            self.assertIn('resolution', message)
            self.assertIn("540p' or '720p", message)
