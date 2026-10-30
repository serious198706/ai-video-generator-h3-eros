"""Bounded Linux/CUDA diagnostics. No model weights, credentials, URLs or AWS access."""
import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from h3_service.infer.cpu_budget import configure_environment, configure_torch


def command(args, *, stdout_limit=40000):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'returncode': result.returncode, 'stdout': result.stdout[:stdout_limit],
                'stdout_truncated': len(result.stdout) > stdout_limit, 'stderr': result.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': type(exc).__name__}


def gpu_info():
    result = command(['nvidia-smi', '-q', '-x'], stdout_limit=1024**2)
    if result.get('returncode') != 0:
        return result
    if result.get('stdout_truncated'):
        return {'error': 'nvidia-smi XML exceeded 1 MiB capture limit'}
    try:
        root = ET.fromstring(result['stdout'])
    except ET.ParseError:
        return {'error': 'nvidia-smi XML invalid or exceeded bounded capture'}
    def fields(node):
        # Preserve nesting: link_gen/current_link_gen and link_widths/current_link_width
        # must not overwrite their maximum values in a flat dictionary.
        return {child.tag: fields(child) if len(child) else child.text for child in node}

    output = {'driver_version': root.findtext('driver_version'), 'gpus': []}
    for gpu in root.findall('gpu'):
        entry = {'name': gpu.findtext('product_name'), 'performance_state': gpu.findtext('performance_state')}
        for section in ('pci', 'fb_memory_usage', 'utilization', 'gpu_power_readings', 'power_readings', 'clocks'):
            node = gpu.find(section)
            if node is not None:
                entry[section] = fields(node)
        output['gpus'].append(entry)
        bus = entry.get('pci', {}).get('pci_bus_id')
        if bus:
            device = Path('/sys/bus/pci/devices') / ('0000:' + bus.split(':', 1)[1])
            entry['sysfs'] = {}
            for name in ('numa_node', 'local_cpulist', 'current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width'):
                try:
                    entry['sysfs'][name] = (device / name).read_text().strip()
                except OSError:
                    entry['sysfs'][name] = 'unavailable'
    return output


def host_info(pid):
    output = {'python': platform.python_version(), 'platform': platform.platform(),
              'cpu_count': os.cpu_count(), 'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    files = ['/proc/meminfo', '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu.stat',
             '/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.events',
             '/sys/fs/cgroup/memory.stat', '/sys/fs/cgroup/cpuset.cpus.effective', '/sys/fs/cgroup/cpuset.mems.effective']
    if pid:
        files += [f'/proc/{pid}/status', f'/proc/{pid}/smaps_rollup', f'/proc/{pid}/io']
    for name in files:
        try:
            output[name] = Path(name).read_text()[:20000]
        except OSError:
            output[name] = 'unavailable'
    output['numa_nodes'] = {str(p): p.read_text()[:10000]
        for pattern in ('node*/cpulist', 'node*/distance')
        for p in Path('/sys/devices/system/node').glob(pattern)}
    output['lscpu'] = command(['lscpu'])
    output['gpu_topology'] = command(['nvidia-smi', 'topo', '-m'])
    output['numactl'] = command(['numactl', '--hardware'])
    output['nvcc'] = command(['nvcc', '--version'])
    return output


def bench_copies():
    configure_environment()
    import torch
    report = {'cpu_threads': configure_torch(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'compiled_arches': torch.cuda.get_arch_list()}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    free, total = torch.cuda.mem_get_info()
    if total - free > 4 * 1024**3:
        raise RuntimeError('GPU is occupied: wait for the current task to finish; diagnostic did not start benchmarks')
    size = 256 * 1024**2
    report['bytes_per_copy'] = size
    report['copies'] = {}

    def measured(label, function, count=3, size_bytes=size):
        values = []
        for _ in range(count):
            torch.cuda.synchronize()
            started = time.perf_counter()
            function()
            torch.cuda.synchronize()
            values.append(time.perf_counter() - started)
        median = statistics.median(values)
        report['copies'][label] = {'seconds': values, 'median_s': median,
                                    'GiB_per_s': size_bytes / 1024**3 / median}

    for threads in sorted({1, report['cpu_threads']}):
        torch.set_num_threads(threads)
        for pinned in (False, True):
            print(f'Copy probe: threads={threads}, pinned={pinned}', flush=True)
            cpu = torch.empty(size, dtype=torch.uint8, pin_memory=pinned).fill_(1)
            gpu = cpu.to('cuda')
            torch.cuda.synchronize()
            measured(f'threads{threads}_pinned{pinned}_H2D_preallocated', lambda: gpu.copy_(cpu))
            measured(f'threads{threads}_pinned{pinned}_D2H_preallocated', lambda: cpu.copy_(gpu))
            # Includes CPU destination allocation as Module.to(cpu) does.
            measured(f'threads{threads}_pinned{pinned}_D2H_allocating', lambda: gpu.to('cpu'))
            del cpu, gpu
            torch.cuda.empty_cache()
    torch.set_num_threads(report['cpu_threads'])
    # Same total bytes spread across many parameters, to expose per-tensor overhead.
    arrays = [torch.ones(size // 64, dtype=torch.uint8) for _ in range(64)]
    def fragmented():
        targets = [array.to('cuda') for array in arrays]
        torch.cuda.synchronize()
        del targets
    measured('64_tensors_H2D_allocating', fragmented)
    del arrays
    from h3_service.infer.cpu_weights import validate_cuda
    report['cpu_weight_restore_cuda_check'] = validate_cuda()
    report['gpu_after_copies'] = gpu_info()  # Snapshot, not a sustained-load link-speed measurement.
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pid', type=int, help='optional current resident PID for RSS/NUMA context; no environment or command line collected')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = {'host_before': host_info(args.pid), 'gpu_before': gpu_info()}
    packages = ('torch', 'torchvision', 'transformers', 'diffusers', 'accelerate', 'safetensors', 'numpy')
    report['packages'] = {}
    for name in packages:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = 'missing'
    try:
        report['benchmarks'] = bench_copies()
        report['status'] = 'succeeded'
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__, reason=str(exc)[:1000])
    finally:
        report['host_after'] = host_info(args.pid)
        path = args.output / 'report.json'
        path.write_text(json.dumps(report, indent=2))
        path.chmod(0o600)
        print(f'Diagnostic status={report["status"]}; report={path}', flush=True)
    return 0 if report['status'] == 'succeeded' else 1


if __name__ == '__main__':
    raise SystemExit(main())
