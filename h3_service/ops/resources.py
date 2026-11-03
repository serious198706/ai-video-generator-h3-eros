"""Pause admission under storage/RAM pressure without unloading healthy models."""
import os
import shutil


def warnings(settings):
    result = []
    if shutil.disk_usage(settings.work_dir).free / 1024**3 < float(os.getenv('H3_MIN_DISK_GIB', '5')):
        result.append('disk_space_low')
    if os.path.exists('/proc/meminfo'):
        from h3_service.infer.resident import available_memory
        if available_memory() / 1024**3 < float(os.getenv('H3_MIN_RAM_GIB', '4')):
            result.append('host_memory_low')
    return result
