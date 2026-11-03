import os
import time


def cleanup_reports(root, now=None):
    now = time.time() if now is None else now
    days = int(os.getenv('H3_REPORT_RETENTION_DAYS', '30'))
    limit = int(os.getenv('H3_REPORT_MAX_MB', '500')) * 1024**2
    paths = sorted(root.glob('*/*.json'), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in paths)
    for path in paths:
        stat = path.stat()
        age = now - stat.st_mtime
        if age > days * 86400 or (total > limit and age > 86400):
            path.unlink(missing_ok=True)
            total -= stat.st_size
    for directory in root.glob('*'):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
