"""Check the Vast host and wheel runtime before downloading model weights.

nvcc is informational: this service does not build CUDA extensions.
The CUDA 13.x driver-family lower bound is R580; actual kernel checks still apply.
"""
import argparse
import json
import subprocess
import sys


def validate_drivers(versions):
    if not versions:
        raise RuntimeError('nvidia-smi reported no GPUs')
    for version in versions:
        try:
            major = int(version.split('.')[0])
        except ValueError:
            raise RuntimeError('cannot determine NVIDIA driver version') from None
        if major < 580:
            raise RuntimeError('cu130 requires NVIDIA driver R580 or newer; select/update the Vast host driver')


def validate_wheel(torch_version, cuda_version, vision_version):
    if torch_version != '2.11.0+cu130' or cuda_version != '13.0' or vision_version != '0.26.0+cu130':
        raise RuntimeError('H3 requires torch 2.11.0+cu130 / torchvision 0.26.0+cu130; use the isolated H3 venv')


def check_host():
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=15)
    versions = sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))
    validate_drivers(versions)
    return {'driver_versions': versions}


def check_runtime(attention='_native_cudnn'):
    report = check_host()
    from h3_service.infer.cpu_budget import configure_environment, configure_torch
    configure_environment()
    import torch
    report['cpu_threads'] = configure_torch()
    import torchvision
    validate_wheel(torch.__version__, torch.version.cuda, torchvision.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the H3 process')
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    with torch.inference_mode():
        matrix = torch.ones((64, 64), device='cuda', dtype=torch.bfloat16)
        if not torch.isfinite(matrix @ matrix).all().item():
            raise RuntimeError('BF16 matrix multiplication failed')
        if attention == '_native_cudnn':
            from torch.nn.attention import sdpa_kernel, SDPBackend
            q = torch.randn(1, 2, 64, 64, device='cuda', dtype=torch.bfloat16)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                output = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            if not torch.isfinite(output).all().item():
                raise RuntimeError('cuDNN attention produced non-finite output')
        torch.cuda.synchronize()
    report.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_runtime=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), device=props.name,
                  compute_capability=list(torch.cuda.get_device_capability(0)),
                  total_memory_gib=props.total_memory / 1024**3, attention=attention,
                  smoke_test='passed', model_generation_tested=False)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.host_only:
            report = check_host()
        else:
            from h3_service.config import Settings
            report = check_runtime(Settings().attention)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(f'H3 runtime check failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
