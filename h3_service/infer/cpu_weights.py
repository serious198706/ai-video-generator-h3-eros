"""Keep frozen CPU parameter storage; release GPU copies without copying weights back.

Inference only: mutable registered buffers are copied back. Runtime weight edits,
training, compilation and dynamic LoRA changes are not supported by this owner.
"""
class CpuWeights:
    def __init__(self, pipe):
        import torch
        self.slots = []
        self.modules = []
        self.parameters = {}
        seen = set()
        for component in pipe.components.values():
            if not isinstance(component, torch.nn.Module):
                continue
            for module in component.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                self.modules.append(module)
                for name, parameter in module._parameters.items():
                    if parameter is None:
                        continue
                    if parameter.device.type != 'cpu' or parameter.grad is not None:
                        raise RuntimeError('CPU weight owner requires CPU inference parameters')
                    key = id(parameter)
                    self.parameters.setdefault(key, (parameter.detach(), parameter.requires_grad))
                    self.slots.append((module, name, key))
        self.bytes = sum(t.numel() * t.element_size() for t, _ in self.parameters.values())
        if not self.slots:
            raise RuntimeError('pipeline has no CPU parameters to retain')

    def restore(self):
        import torch
        import time
        started = time.monotonic()
        restored = {}
        for module, name, key in self.slots:
            tensor, requires_grad = self.parameters[key]
            current = module._parameters[name]
            if current.shape != tensor.shape or current.dtype != tensor.dtype or current.grad is not None:
                raise RuntimeError('model parameter layout changed; CPU restore refused')
            if key not in restored:
                restored[key] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
            module._parameters[name] = restored[key]
        parameter_s = time.monotonic() - started
        # Buffers can be mutable (e.g. rotary/cache state), so preserve their current values.
        buffers = {}
        started = time.monotonic()
        for module in self.modules:
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    if id(buffer) not in buffers:
                        buffers[id(buffer)] = buffer.to('cpu')
                    module._buffers[name] = buffers[id(buffer)]
        return {'parameter_restore_s': parameter_s, 'buffer_restore_s': time.monotonic() - started,
                'retained_parameter_gib': self.bytes / 1024**3}


def validate_cuda():
    """Small actual-device check before loading expensive models."""
    import torch
    from types import SimpleNamespace
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            self.right = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            self.right.weight = self.left.weight
            self.register_buffer('scale', torch.tensor(1.0))
        def forward(self, x):
            return self.right(self.left(x)) * self.scale
    model = Fixture().eval()
    owner = CpuWeights(SimpleNamespace(components={'model': model}))
    x = torch.ones(4, 64, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        model.to('cuda')
        model.scale.add_(1)
        expected = model(x).cpu()
        for _ in range(3):
            owner.restore()
            if model.left.weight is not model.right.weight or model.scale.item() != 2:
                raise RuntimeError('CPU restore alias or buffer validation failed')
            model.to('cuda')
            torch.testing.assert_close(model(x).cpu(), expected, rtol=0, atol=0)
    owner.restore()
    del owner, model, x
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return 'passed: tied parameters, mutable buffer, three cycles, exact GPU output'
