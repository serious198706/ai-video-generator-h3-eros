import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from h3_service.config import Settings
from h3_service.infer.resident import ResidentModels, ResidentEngine, available_memory
from h3_service.infer.engine import LeaseLost


class ResidentTests(unittest.TestCase):
    def test_container_limit_and_reclaimable_cache_bound_ram(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'meminfo').write_text('MemAvailable: 894435328 kB\n')
            (root / 'memory.max').write_text(str(342 * 1024**3))
            (root / 'memory.current').write_text(str(180 * 1024**3))
            (root / 'memory.stat').write_text(f'inactive_file {40 * 1024**3}\n')
            self.assertEqual(available_memory(root / 'meminfo', root), 202 * 1024**3)
            (root / 'meminfo').write_text('MemAvailable: 104857600 kB\n')
            self.assertEqual(available_memory(root / 'meminfo', root), 100 * 1024**3)

    def test_consecutive_jobs_load_twice_and_convert_once(self):
        import torch
        pipes = [MagicMock(), MagicMock()]
        for pipe in pipes:
            pipe.components = {'model': torch.nn.Linear(2, 2, bias=False)}
        blocks = MagicMock()
        blocks.load_pipeline.side_effect = pipes
        eros = MagicMock()
        eros.apply_eros.return_value = {'replaced': 634}
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict('sys.modules', {'h3_service.infer.blocks': blocks, 'h3_service.infer.eros': eros}), \
                patch('h3_service.infer.resident.available_memory', return_value=220 * 1024**3), \
                patch('torch.cuda.mem_get_info', return_value=(94 * 1024**3, 95 * 1024**3)), \
                patch('torch.cuda.synchronize'), patch('torch.cuda.empty_cache'), \
                patch('h3_service.infer.cpu_weights.validate_cuda', return_value='passed'), \
                patch('h3_service.infer.phase.execute', return_value={'load_s': 0}) as execute:
            models = ResidentModels(Settings(converted_cache=False), lambda message: None)
            for _ in range(2):
                for mode in ('condition', 'generate'):
                    result = models.run(mode, {'directory': temp})
                    self.assertTrue(result['resident'])
            self.assertEqual(blocks.load_pipeline.call_count, 2)
            eros.apply_eros.assert_called_once()
            self.assertEqual(execute.call_count, 4)
            self.assertIs(execute.call_args_list[0].kwargs['loaded_pipe'], pipes[0])
            self.assertIs(execute.call_args_list[2].kwargs['loaded_pipe'], pipes[0])
            for pipe in pipes:
                self.assertEqual([c.args[0] for c in pipe.to.call_args_list], ['cuda'])

    def test_insufficient_container_memory_fails_before_loading(self):
        blocks = MagicMock()
        eros = MagicMock()
        with patch.dict('sys.modules', {'h3_service.infer.blocks': blocks, 'h3_service.infer.eros': eros}), \
                patch('h3_service.infer.resident.available_memory', return_value=100 * 1024**3):
            with self.assertRaisesRegex(RuntimeError, 'RAM'):
                ResidentModels(Settings(), lambda message: None)
        blocks.load_pipeline.assert_not_called()

    def test_cancelled_wait_kills_resident_and_clears_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = ResidentEngine(Settings())
            engine.healthy = True
            engine.process = MagicMock()
            engine.process.poll.return_value = None
            engine.process.pid = 123
            engine.ipc = MagicMock(name='ipc')
            engine.ipc.name = temp
            cancelled = threading.Event()
            cancelled.set()
            with patch('os.killpg') as kill:
                with self.assertRaises(LeaseLost):
                    engine.generate({}, Path(temp), cancelled, lambda stage: None)
                kill.assert_called_once()
            self.assertFalse(engine.healthy)
            self.assertIsNone(engine.process)

    def test_dead_model_process_is_not_reused(self):
        engine = ResidentEngine(Settings())
        engine.process = MagicMock()
        engine.process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'resident_process_exited'):
                engine.wait_reply(Path(temp) / 'missing.json', threading.Event())
