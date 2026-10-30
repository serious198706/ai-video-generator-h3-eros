import unittest
from types import SimpleNamespace
import torch
from h3_service.infer.cpu_weights import CpuWeights


class CpuWeightsTests(unittest.TestCase):
    def test_meta_device_roundtrip_preserves_cpu_storage_values_and_parameter_alias(self):
        model = torch.nn.Module()
        model.left = torch.nn.Linear(3, 3, bias=False)
        model.right = torch.nn.Linear(3, 3, bias=False)
        model.right.weight = model.left.weight
        expected = model.left.weight.detach().clone()
        pointer = model.left.weight.data_ptr()
        owner = CpuWeights(SimpleNamespace(components={'model': model}))
        for _ in range(3):
            model.to('meta')  # Represents releasing CPU ownership during a device conversion.
            owner.restore()
            torch.testing.assert_close(model.left.weight, expected)
            self.assertEqual(model.left.weight.data_ptr(), pointer)
            self.assertIs(model.left.weight, model.right.weight)

    def test_layout_changes_fail_instead_of_restoring_wrong_weights(self):
        model = torch.nn.Linear(2, 2, bias=False)
        owner = CpuWeights(SimpleNamespace(components={'model': model}))
        model.weight = torch.nn.Parameter(torch.ones(3, 3))
        with self.assertRaises(RuntimeError):
            owner.restore()

    def test_mutable_registered_buffer_value_is_preserved(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer('counter', torch.tensor(1))
        owner = CpuWeights(SimpleNamespace(components={'model': model}))
        model.counter.add_(4)
        owner.restore()
        self.assertEqual(model.counter.item(), 5)
