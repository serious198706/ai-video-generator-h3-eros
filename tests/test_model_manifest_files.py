import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from h3_service.download_models import manifest_files


class ManifestFilesTests(unittest.TestCase):
    def test_transfer_artifacts_do_not_become_required_model_files(self):
        with tempfile.TemporaryDirectory(prefix='.h3-') as directory:
            root = Path(directory)
            assets = ['text_encoder/model-00003.safetensors', 'tokenizer/vocab.json',
                      'modular_model_index.json']
            artifacts = ['text_encoder/.model-00003.safetensors.3yJeIu',
                         '.cache/huggingface/file.metadata', 'text_encoder/file.partial',
                         'text_encoder/file.incomplete', 'text_encoder/file.tmp',
                         'text_encoder/file.lock', 'h3-manifest.json']
            for name in assets + artifacts:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual({str(p.relative_to(root)) for p in manifest_files(root)}, set(assets))

    def test_startup_missing_file_log_identifies_path_and_call_site(self):
        from h3_service.config import Settings
        from h3_service.tasks.worker import Worker
        engine = MagicMock()
        engine.preflight.side_effect = FileNotFoundError(2, 'No such file', '/models/base/missing.json')
        worker = Worker(Settings(), MagicMock(), MagicMock(), engine)
        with patch('h3_service.aws_setup.verify_credentials'), patch('h3_service.aws_setup.verify_queue'):
            with self.assertLogs('h3_service.tasks.worker', level='ERROR') as logs:
                worker.run()
        self.assertIn('/models/base/missing.json', logs.output[0])
        self.assertIn('"trace"', logs.output[0])
        self.assertFalse(worker.ready)
