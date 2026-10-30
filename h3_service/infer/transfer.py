"""Observe actual placement, including CPU work and file-backed weight faults."""
import json
import resource
import time
from pathlib import Path


def counters():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {'major_faults': usage.ru_majflt, 'minor_faults': usage.ru_minflt,
              'cpu_user_s': usage.ru_utime, 'cpu_system_s': usage.ru_stime}
    try:
        io = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
        result.update(read_bytes=int(io['read_bytes']), read_syscalls=int(io['syscr']))
    except (OSError, KeyError, ValueError):
        pass
    return result


def place_on_gpu(pipe, emit):
    import torch
    before = counters()
    started = time.monotonic()
    pipe.to('cuda')
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    after = counters()
    result = {key: value - before[key] for key, value in after.items() if key in before}
    result['wall_s'] = elapsed
    emit(f'GPU placement complete timing={json.dumps(result)}')
    return result
