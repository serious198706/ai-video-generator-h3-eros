"""Download and verify the compact SR weight for local dual-GPU inference."""
import hashlib
import os
import sys
import urllib.request
from h3_service.config import Settings

SHA256 = '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292'
URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'


def main():
    path = Settings().sr_model
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.download')
        try:
            urllib.request.urlretrieve(URL, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
                raise ValueError('SR checkpoint SHA256 mismatch')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('SR checkpoint SHA256 mismatch')
    print('Compact SR checkpoint verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
