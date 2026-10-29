import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from h3_service.aws_setup import bootstrap, save_credentials
from h3_service.config import Settings
from h3_service.infer.manifest import verify, sha256, BASE_REVISION, COMPONENTS
from h3_service.infer.converted_cache import load_generator


class StartupCacheTests(unittest.TestCase):
    def test_credentials_atomic_persistence_preserves_config_and_removes_stale_token(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'fake id', 'AWS_SECRET_ACCESS_KEY': "fake' $() secret"}, clear=True):
            file = Path(temp) / '.env'
            file.write_text('H3_QUEUE_URL=https://example.com\nexport AWS_SESSION_TOKEN=stale\nAWS_SECRET_ACCESS_KEY=old\n')
            save_credentials(file, 'us-east-1')
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)
            self.assertIn('H3_QUEUE_URL=', file.read_text())
            self.assertNotIn('AWS_SESSION_TOKEN=', file.read_text())
            result = subprocess.run(['bash', '-c', 'source "$1"; test "$AWS_SECRET_ACCESS_KEY" = "$2"',
                'test', str(file), os.environ['AWS_SECRET_ACCESS_KEY']], check=False)
            self.assertEqual(result.returncode, 0)

    def test_invalid_credentials_never_overwrite_file(self):
        with tempfile.TemporaryDirectory() as temp, patch('boto3.Session'), \
                patch('h3_service.aws_setup.verify_credentials', side_effect=RuntimeError):
            file = Path(temp) / '.env'
            file.write_text('keep\n')
            with self.assertRaises(RuntimeError):
                bootstrap('us-east-1', save=file)
            self.assertEqual(file.read_text(), 'keep\n')

    def test_verified_unchanged_weights_skip_hash_then_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in COMPONENTS:
                (root / name).mkdir()
            index = root / 'modular_model_index.json'
            index.write_text('{}')
            eros = root / 'eros'
            eros.write_text('weights')
            (root / 'h3-manifest.json').write_text(json.dumps({'revision': BASE_REVISION,
                'files': {'modular_model_index.json': sha256(index)}}))
            settings = Settings(model_dir=root, eros_file=eros)
            cache = root / 'verified.json'
            with patch('h3_service.infer.manifest.EROS_SHA256', sha256(eros)):
                verify(settings, cache_path=cache)
                with patch('h3_service.infer.manifest.sha256', side_effect=AssertionError('unexpected reread')):
                    verify(settings, cache_path=cache)
                index.write_text('bad')
                with self.assertRaises(ValueError):
                    verify(settings, cache_path=cache)

    def test_converted_checkpoint_reused_and_corruption_rebuilds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipe, loader, converter = MagicMock(), MagicMock(), MagicMock()
            loader.return_value = pipe
            converter.return_value = {'replaced': 634}
            def save(path, **kwargs):
                (path / 'config.json').write_text('{}')
                (path / 'model.safetensors').write_bytes(b'fake-test-weights')
            pipe.transformer.save_pretrained.side_effect = save
            settings = Settings(transformer_cache=root)
            load_generator(settings, lambda x: None, loader, converter)
            load_generator(settings, lambda x: None, loader, converter)
            converter.assert_called_once()
            self.assertIn('transformer_path', loader.call_args.kwargs)
            cached = next(root.glob('eros-*'))
            (cached / 'model.safetensors').write_bytes(b'corrupt')
            load_generator(settings, lambda x: None, loader, converter)
            self.assertEqual(converter.call_count, 2)
