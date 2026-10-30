import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from h3_service.config import ROOT


def setup_logging():
    directory = Path(os.getenv('H3_LOG_DIR', str(ROOT / 'logs')))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / 'h3.log'
    file = TimedRotatingFileHandler(path, when='midnight', backupCount=30, encoding='utf-8')
    os.chmod(path, 0o600)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s',
                        handlers=[logging.StreamHandler(), file], force=True)
