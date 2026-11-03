import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path
from h3_service.config import ROOT, Settings


def configuration(root, directory, python, grace, entry=None):
    # Supervisor expands percent signs and treats semicolons as comments.
    entry = Path(entry) if entry else root / 'start.sh'
    for path in (root, directory, python, entry):
        if any(c in str(path) for c in '\n\r%;'):
            raise ValueError('service paths cannot contain newline, percent or semicolon')
    return f'''[unix_http_server]
file={directory}/supervisor.sock
chmod=0600
[supervisord]
logfile={directory}/supervisord.log
logfile_maxbytes=5MB
logfile_backups=3
pidfile={directory}/supervisord.pid
childlogdir={directory}
umask=0077
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{directory}/supervisor.sock
[program:h3]
directory={root}
command=/bin/bash {shlex.quote(str(entry))} --foreground
environment=H3_MANAGED="1"
autostart=true
autorestart=unexpected
exitcodes=0,78
startsecs=5
startretries=3
stopsignal=TERM
stopwaitsecs={grace + 60}
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={directory}/console.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=4
'''


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description='Manage the private H3 Supervisor; never controls Vast services.')
    parser.add_argument('action', choices=['install', 'start', 'stop', 'restart', 'status', 'logs', 'reset', 'shutdown'])
    args = parser.parse_args()
    control = './' + Path(os.getenv('H3_SERVICE_ENTRY', 'service.sh')).name
    settings = Settings()
    directory = settings.work_dir.resolve() / 'service'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    config = directory / 'supervisord.conf'
    ctl = [sys.executable, '-m', 'supervisor.supervisorctl', '-c', str(config)]
    try:
        import supervisor  # noqa: F401
    except ImportError:
        parser.error('Supervisor missing; run ./deploy.sh to install dependencies')
    with (directory / 'manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'install':
            text = configuration(ROOT, directory, Path(sys.executable), settings.shutdown_grace,
                                 entry=os.getenv('H3_SERVICE_ENTRY'))
            if config.exists() and config.read_text() != text:
                alive = subprocess.run(ctl + ['pid'], capture_output=True).returncode == 0
                if alive:
                    parser.error(f'Supervisor is running with another configuration; {control} shutdown before reinstalling')
            temporary = config.with_suffix('.tmp')
            temporary.write_text(text)
            temporary.chmod(0o600)
            temporary.replace(config)
            if (ROOT / '.env').exists():
                (ROOT / '.env').chmod(0o600)
            print(f'Installed {config}. No service started or restarted. Use {control} start.')
            return 0
        if not config.exists():
            parser.error(f'Run {control} install first')
        if args.action == 'reset':
            (directory / 'failures.json').unlink(missing_ok=True)
            print(f'Automatic restart budget reset. Use {control} start after fixing the cause.')
            return 0
        alive = subprocess.run(ctl + ['pid'], capture_output=True).returncode == 0
        if args.action in {'start', 'restart'}:
            if not alive:
                with (settings.work_dir / 'worker.lock').open('a') as worker_lock:
                    try:
                        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        parser.error('Another H3 worker is running; stop its foreground process before managed startup')
                # Credentials are read fresh by start.sh; never retain them in the daemon environment.
                environment = {k: v for k, v in os.environ.items() if k not in {
                    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_SECURITY_TOKEN'}}
                subprocess.run([sys.executable, '-m', 'supervisor.supervisord', '-c', str(config)],
                               check=True, env=environment)
                # autostart owns initial launch; do not send restart during loading.
            else:
                if args.action == 'start':
                    status = subprocess.run(ctl + ['status', 'h3'], capture_output=True, text=True)
                    if any(state in status.stdout for state in ('RUNNING', 'STARTING')):
                        print(status.stdout.strip() + '; existing process retained')
                        return 0
                result = subprocess.run(ctl + [args.action, 'h3'])
                if result.returncode:
                    return result.returncode
            print(f'Process launch requested. RUNNING does not mean model ready; use {control} status.')
            return 0
        if not alive:
            print('H3 Supervisor is not running.')
            return 1 if args.action == 'status' else 0
        action = {'logs': ['tail', '-f', 'h3'], 'status': ['status', 'h3'],
                  'stop': ['stop', 'h3'], 'shutdown': ['shutdown']}[args.action]
        if args.action == 'logs':
            fcntl.flock(lock, fcntl.LOCK_UN)
        code = subprocess.run(ctl + action).returncode
        if args.action == 'status':
            try:
                url = 'http://127.0.0.1:' + os.getenv('H3_PORT', '8001') + '/health'
                with urllib.request.urlopen(url, timeout=3) as response:
                    health = json.load(response)
                    print(json.dumps(health, ensure_ascii=False, indent=2))
                    if health.get('error'):
                        code = 1
                    elif not health.get('model_ready') or not health.get('accepting_tasks', True):
                        code = code or 2
            except (OSError, ValueError) as exc:
                print(f'Health endpoint unavailable: {type(exc).__name__}')
                code = code or 1
        return code


if __name__ == '__main__':
    raise SystemExit(main())
