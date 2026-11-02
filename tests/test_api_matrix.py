import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class MatrixTests(unittest.TestCase):
    def test_nine_posts_and_resume_without_duplicate_submissions(self):
        spec = importlib.util.spec_from_file_location(
            'matrix', Path(__file__).resolve().parents[1] / 'scripts/benchmark_api_matrix.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        session = MagicMock()
        posted = []
        def post(url, **kwargs):
            posted.append(kwargs['json'])
            response = MagicMock(status_code=202)
            response.json.return_value = {'task_id': f'{len(posted):032x}'}
            return response
        session.post.side_effect = post
        response = MagicMock()
        response.json.return_value = {'status': 'succeeded', 'stage': 'complete',
                                      'timing': {'generate': {'wall_s': 42}, 'upscale': {'wall_s': 12}},
                                      'result': {'width': 1920, 'height': 1080}}
        session.get.return_value = response
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / 'prompt.txt'
            prompt.write_text('a forest')
            arguments = ['matrix', '--api-base', 'https://api.example.com',
                         '--image-url', 'https://cdn.example.com/image.jpg',
                         '--prompt-file', str(prompt), '--output', str(root / 'report')]
            with patch('sys.argv', arguments), patch.object(module.requests, 'Session', return_value=session):
                self.assertEqual(module.main(), 0)
                self.assertEqual(module.main(), 0)
            self.assertEqual(len(posted), 9)
            self.assertEqual({(p['duration'], p['resolution']) for p in posted},
                             {(d, r) for d in (5, 10, 15) for r in ('480p', '720p', '1080p')})
            report = json.loads((root / 'report/report.json').read_text())
            self.assertEqual(report['cases'][0]['generation_s'], 42)
            self.assertEqual(report['cases'][0]['upscale_s'], 12)
