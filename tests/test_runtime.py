import unittest
from types import SimpleNamespace
from unittest.mock import patch
from h3_service.check_runtime import check_host, validate_drivers, validate_wheel


class RuntimeTests(unittest.TestCase):
    def test_compatible_driver_families(self):
        validate_drivers(['580.65.06', '595.0'])

    def test_old_missing_or_unknown_driver_rejected(self):
        for versions in ([], ['570.124.06'], ['N/A']):
            with self.subTest(versions=versions), self.assertRaises(RuntimeError):
                validate_drivers(versions)

    def test_cu128_does_not_pass_version_check(self):
        with self.assertRaises(RuntimeError):
            validate_wheel('2.11.0+cu128', '12.8', '0.26.0+cu128')

    def test_cpu_or_wrong_runtime_rejected(self):
        for version, cuda in [('2.11.0', None), ('2.11.0+cu130', '12.8')]:
            with self.subTest(version=version, cuda=cuda), self.assertRaises(RuntimeError):
                validate_wheel(version, cuda, '0.26.0+cu130')

    def test_expected_wheel(self):
        validate_wheel('2.11.0+cu130', '13.0', '0.26.0+cu130')

    def test_host_check_does_not_require_nvcc(self):
        with patch('subprocess.run', return_value=SimpleNamespace(stdout='580.65.06\n580.65.06\n')) as run:
            self.assertEqual(check_host(), {'driver_versions': ['580.65.06']})
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][0], 'nvidia-smi')
