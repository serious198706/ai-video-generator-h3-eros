import unittest
from pydantic import ValidationError
from h3_service.contracts.models import GenerateRequest, Message, frame_count


class ContractTests(unittest.TestCase):
    def test_frames(self):
        self.assertEqual(frame_count(5), 124)
        self.assertEqual(frame_count(2), 56)
        self.assertEqual(frame_count(14), 345)
        self.assertEqual(frame_count(5.001), 124)

    def test_reject_unsupported(self):
        for value in ({'duration': 15}, {'duration': float('nan')}, {'resolution': '1080p'},
                      {'negativePrompt': 'blur'}, {'quality': 6}, {'steps': 3}, {'steps': True},
                      {'image': ' '}, {'prompt': '  '}, {'audio': 'false'}, {'seed': -1}, {'unknown': 1}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                GenerateRequest.model_validate({'image': 'https://example.com/image', 'prompt': 'a scene', **value})

    def test_aliases(self):
        request = GenerateRequest(image=' https://example.com/image ', prompt=' a scene ', audio=None,
                                  lastImage='https://example.com/last', negativePrompt=' ')
        self.assertFalse(request.audio)
        self.assertEqual(request.prompt, 'a scene')
        self.assertEqual(request.last_image, 'https://example.com/last')

    def test_id_is_not_path(self):
        with self.assertRaises(ValidationError):
            Message(task_id='../other', request={'image': 'x', 'prompt': 'y'})
