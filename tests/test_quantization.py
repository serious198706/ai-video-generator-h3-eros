import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from h3_service.infer.quantization import GENERATOR_BF16_PREFIXES, int8wo_config, int8wo_filter


class QuantizationTests(unittest.TestCase):
    def test_generator_keeps_boundary_blocks_and_quantizes_inner_linear_layers(self):
        linear = torch.nn.Linear(128, 128)
        for prefix in GENERATOR_BF16_PREFIXES:
            self.assertFalse(int8wo_filter('generator', linear, prefix + 'attn.to_q'))
        self.assertTrue(int8wo_filter('generator', linear, 'transformer_blocks.2.attn.to_q'))
        self.assertTrue(int8wo_filter('generator', linear, 'token_refiner.refiner_blocks.0.attn.to_q'))

    def test_conditioner_keeps_lm_head_and_only_selects_linear(self):
        self.assertFalse(int8wo_filter('conditioner', torch.nn.Linear(128, 128), 'lm_head'))
        self.assertTrue(int8wo_filter('conditioner', torch.nn.Linear(128, 128), 'model.layers.0.self_attn.q_proj'))
        self.assertFalse(int8wo_filter('conditioner', torch.nn.LayerNorm(128), 'model.norm'))

    def test_group_incompatible_linear_stays_bf16(self):
        self.assertFalse(int8wo_filter('conditioner', torch.nn.Linear(126, 128), 'visual.proj'))

    def test_unknown_component_is_rejected(self):
        with self.assertRaises(ValueError):
            int8wo_filter('other', torch.nn.Linear(128, 128), 'x')

    def test_torchao_v2_uses_per_group_granularity(self):
        calls = {}
        class PerGroup:
            def __init__(self, size):
                self.size = size
        def config(**kwargs):
            calls.update(kwargs)
            return SimpleNamespace(**kwargs)
        fake = SimpleNamespace(Int8WeightOnlyConfig=config, PerGroup=PerGroup)
        with patch.dict('sys.modules', {'torchao': SimpleNamespace(), 'torchao.quantization': fake}):
            result = int8wo_config()
        self.assertEqual(result.version, 2)
        self.assertEqual(result.granularity.size, 128)
        self.assertNotIn('group_size', calls)
