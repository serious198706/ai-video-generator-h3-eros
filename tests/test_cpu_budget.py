import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from h3_service.infer.cpu_budget import thread_budget, configure_environment


class CpuBudgetTests(unittest.TestCase):
    def test_vast_quota_not_96_visible_cpus(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True), \
                patch('os.sched_getaffinity', return_value=set(range(96)), create=True):
            file = Path(temp) / 'cpu.max'
            file.write_text('1152000 100000')
            self.assertEqual(thread_budget(file), 8)
            os.environ['H3_CPU_THREADS'] = '96'
            self.assertEqual(thread_budget(file), 11)
            file.write_text('150000 100000')
            self.assertEqual(thread_budget(file), 1)

    def test_thread_pool_environment_is_bounded(self):
        with patch.dict(os.environ, {}, clear=True), patch('h3_service.infer.cpu_budget.thread_budget', return_value=8):
            configure_environment()
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '8')
            self.assertEqual(os.environ['MKL_NUM_THREADS'], '8')
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '8')
