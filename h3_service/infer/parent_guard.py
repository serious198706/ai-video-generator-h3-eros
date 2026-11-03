"""Linux model owners die with the API, including an uncatchable API SIGKILL."""
import os
import signal
import sys


def protect_parent():
    expected = os.getenv('H3_OWNER_PID')
    if not expected or sys.platform != 'linux':
        return
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'parent death guard setup failed')
    if os.getppid() != int(expected):
        raise RuntimeError('model owner parent already exited')
