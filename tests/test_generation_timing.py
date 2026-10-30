import sys
import types
import unittest
from unittest.mock import patch
from h3_service.infer.timing import generation_blocks


class GenerationTimingTests(unittest.TestCase):
    def test_scoped_inherited_block_calls_restore_on_failure(self):
        class Parent:
            def __call__(self, fail=False):
                if fail:
                    raise ValueError('test')
                return 'state'
        class Core(Parent):
            pass
        class Video(Parent):
            pass
        class Audio(Parent):
            pass
        blocks = types.ModuleType('blocks')
        blocks.MiniMaxH3CoreDenoiseStep = Core
        decoders = types.ModuleType('decoders')
        decoders.MiniMaxH3VideoDecodeStep = Video
        decoders.MiniMaxH3AudioDecodeStep = Audio
        modules = {'diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3': blocks,
                   'diffusers.modular_pipelines.minimax_h3.decoders': decoders}
        timings, messages = {}, []
        with patch.dict(sys.modules, modules), patch('torch.cuda.synchronize'):
            with self.assertRaises(ValueError):
                with generation_blocks(messages.append, timings):
                    self.assertEqual(Core()(), 'state')
                    self.assertEqual(Video()(), 'state')
                    Audio()(fail=True)
        self.assertEqual(set(timings), {'denoise_s', 'video_decode_s', 'audio_decode_s'})
        self.assertTrue(any('audio_decode: failed' in m for m in messages))
        for cls in (Core, Video, Audio):
            self.assertNotIn('__call__', cls.__dict__)
