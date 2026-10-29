"""Small GPU copy diagnostic; no models, queues, credentials or services touched."""
import json
import statistics
import time
from h3_service.infer.cpu_budget import configure_environment, configure_torch


def main():
    configure_environment()
    import torch
    threads = configure_torch()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    size = 256 * 1024**2
    report = {'torch': torch.__version__, 'cpu_threads': threads, 'bytes_per_copy': size, 'copies': {}}
    # Pre-allocate and touch CPU pages before timing: distinguish memcpy from allocation/page faults.
    for pinned in (False, True):
        cpu = torch.empty(size, dtype=torch.uint8, pin_memory=pinned).fill_(1)
        gpu = torch.empty(size, dtype=torch.uint8, device='cuda')
        gpu.copy_(cpu)
        torch.cuda.synchronize()
        for direction in ('H2D', 'D2H'):
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                (gpu if direction == 'H2D' else cpu).copy_(cpu if direction == 'H2D' else gpu)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
            seconds = statistics.median(times)
            report['copies'][f'{"pinned" if pinned else "pageable"}_{direction}'] = {
                'seconds': seconds, 'GiB_per_s': size / 1024**3 / seconds}
        del cpu, gpu
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
