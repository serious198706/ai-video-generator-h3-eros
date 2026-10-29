"""Bound CPU thread pools by container quota, not host CPU count."""
import math
import os
from pathlib import Path


def thread_budget(quota_file=Path('/sys/fs/cgroup/cpu.max')):
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota, period = quota_file.read_text().split()
        if quota != 'max':
            available = min(available, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    requested = int(os.getenv('H3_CPU_THREADS', '8'))
    if requested < 1:
        raise ValueError('H3_CPU_THREADS must be positive')
    return max(1, min(requested, available))


def configure_environment():
    threads = thread_budget()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    return threads


def configure_torch():
    threads = configure_environment()
    import torch
    torch.set_num_threads(threads)
    return threads
