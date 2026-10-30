import unittest

import torch

from h3_service.infer.quantization import GENERATOR_BF16_PREFIXES, int8wo_filter


class QuantizationTests(unittest.TestCase):
    def test_generator_keeps_boundary_blocks_and_quantizes_inner_linear_layers(self):
        linear = torch.nn.Linear(8, 8)
        for prefix in GENERATOR_BF16_PREFIXES:
            self.assertFalse(int8wo_filter('generator', linear, prefix + 'attn.to_q'))
        self.assertTrue(int8wo_filter('generator', linear, 'transformer_blocks.2.attn.to_q'))
        self.assertTrue(int8wo_filter('generator', linear, 'token_refiner.refiner_blocks.0.attn.to_q'))

    def test_conditioner_keeps_lm_head_and_only_selects_linear(self):
        self.assertFalse(int8wo_filter('conditioner', torch.nn.Linear(8, 8), 'lm_head'))
        self.assertTrue(int8wo_filter('conditioner', torch.nn.Linear(8, 8), 'model.layers.0.self_attn.q_proj'))
        self.assertFalse(int8wo_filter('conditioner', torch.nn.LayerNorm(8), 'model.norm'))

    def test_unknown_component_is_rejected(self):
        with self.assertRaises(ValueError):
            int8wo_filter('other', torch.nn.Linear(8, 8), 'x')
