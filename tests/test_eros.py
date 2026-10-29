import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import save_file
from h3_service.infer.eros import map_tensor, rebuild_adaln, apply_eros, KEEP
from h3_service.infer.phase import validate_wire


class ErosTests(unittest.TestCase):
    def test_qkv(self):
        original = torch.arange(24).reshape(6, 4)
        result = dict(map_tensor('blocks.0.attn.qkv_proj.weight', original, 2))
        self.assertEqual(list(result), [f'transformer_blocks.0.attn.to_{x}.weight' for x in 'qkv'])
        torch.testing.assert_close(result['transformer_blocks.0.attn.to_v.weight'], original[4:])

    def test_gate_value(self):
        source = torch.arange(12).reshape(4, 3)
        key, mapped = map_tensor('token_refiner.blocks.0.mlp.fc1.weight', source, 2)[0]
        self.assertEqual(key, 'token_refiner.refiner_blocks.0.ff.net.0.proj.weight')
        torch.testing.assert_close(mapped, torch.cat([source[2:], source[:2]]))

    def test_adaln_equation(self):
        torch.manual_seed(42)
        basis, mean, weight, bias, x = torch.randn(8, 16), torch.randn(16), torch.randn(32, 8), torch.randn(32), torch.randn(16)
        w, b = rebuild_adaln(weight, bias, basis, mean)
        torch.testing.assert_close(w @ x + b, weight @ (basis @ (x - mean)) + bias, rtol=1e-5, atol=1e-5)

    def test_unknown_untouched_parameter_fails(self):
        params = {k: torch.nn.Parameter(torch.ones(1)) for k in KEEP | {'unexpected'}}
        transformer = SimpleNamespace(named_parameters=lambda: params.items(),
                                      config=SimpleNamespace(num_attention_heads=1, attention_head_dim=2))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'weights.safetensors'
            save_file({'rope.inv_freq': torch.ones(1), 'adaln_basis': torch.ones(8, 16),
                       'adaln_mean': torch.zeros(16), 'adaln_t_table': torch.zeros(1025, 8)}, str(path))
            with self.assertRaisesRegex(ValueError, 'untouched'):
                apply_eros(transformer, path)

    def test_wire_rejects_wrong_dtype_and_shape(self):
        embeds, tags = torch.zeros(1, 2, 5120, dtype=torch.bfloat16), torch.ones(2, dtype=torch.int64)
        validate_wire(embeds, tags)
        for bad_embeds, bad_tags in [(embeds.float(), tags), (embeds, tags[:1]), (embeds, tags * 3)]:
            with self.assertRaises(ValueError):
                validate_wire(bad_embeds, bad_tags)
