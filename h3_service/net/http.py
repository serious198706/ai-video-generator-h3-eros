"""Wan-compatible image transport; separate pinned HTTPS transport for callbacks."""
import ipaddress
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit, urljoin
import urllib3

MAX_IMAGE_BYTES = 25 * 1024 * 1024


class UrlError(ValueError):
    def __init__(self, message, *, http_status=None):
        super().__init__(message)
        self.code = message.replace(' ', '_')
        self.http_status = http_status


def resolve(url, hosts=(), *, webhook=False):
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port or 443
    except ValueError:
        raise UrlError('invalid URL') from None
    if parsed.scheme != 'https' or not host or parsed.username or parsed.password or parsed.fragment:
        raise UrlError('HTTPS URL without credentials or fragment required')
    if port != 443:
        raise UrlError('only HTTPS port 443 is allowed')
    allowed = [h.strip().lower().rstrip('.') for h in hosts if h.strip()]
    host = host.lower().rstrip('.')
    if allowed and not any(host == h or (h.startswith('.') and (host.endswith(h) or host == h[1:])) for h in allowed):
        raise UrlError('host is not allowed')
    try:
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
    except OSError:
        raise UrlError('DNS resolution failed') from None
    if not addresses:
        raise UrlError('no addresses')
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        # Private webhook destinations require an explicit configured host allowlist.
        private_ok = webhook and bool(allowed) and ip.is_private
        if (not ip.is_global and not private_ok) or ip.is_link_local or ip.is_loopback or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            raise UrlError('destination address is not allowed')
    return parsed, addresses


def request(url, *, hosts=(), webhook=False, method='GET', body=None, headers=None):
    parsed, addresses = resolve(url, hosts, webhook=webhook)
    # Connect to the validated IP while retaining hostname for TLS/SNI verification.
    pool = urllib3.HTTPSConnectionPool(addresses[0], port=443, server_hostname=parsed.hostname,
                                     assert_hostname=parsed.hostname, cert_reqs='CERT_REQUIRED', maxsize=1)
    target = parsed.path or '/'
    if parsed.query:
        target += '?' + parsed.query
    try:
        response = pool.urlopen(method, target, body=body,
                                headers={'Host': parsed.hostname, **(headers or {})},
                                timeout=urllib3.Timeout(connect=5, read=10), retries=False,
                                redirect=False, preload_content=False)
        return pool, response
    except Exception:
        pool.close()
        raise


def download(url, destination: Path, hosts=()):
    """Use Wan's requests Session/hostname transport for image downloads."""
    import os
    import requests
    timeout = float(os.getenv('H3_DOWNLOAD_TIMEOUT', os.getenv('WAN22_DOWNLOAD_TIMEOUT', '30')))
    deadline = time.monotonic() + 90
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for _ in range(6):  # Wan permits up to five redirects.
            resolve(url, hosts)  # Validate each redirect before connecting.
            if time.monotonic() > deadline:
                raise UrlError('image download timeout')
            try:
                with session.get(url, stream=True, timeout=timeout, allow_redirects=False,
                                 headers={'User-Agent': 'wan22-server/1'}) as response:
                    if response.is_redirect:
                        location = response.headers.get('Location')
                        if not location:
                            raise UrlError('redirect without location')
                        url = urljoin(url, location)
                        continue
                    if response.status_code != 200:
                        raise UrlError('image download rejected', http_status=response.status_code)
                    if int(response.headers.get('Content-Length', '0')) > MAX_IMAGE_BYTES:
                        raise UrlError('image too large')
                    size = 0
                    with destination.open('wb') as output:
                        for chunk in response.iter_content(chunk_size=65536):
                            if not chunk:
                                continue
                            size += len(chunk)
                            if size > MAX_IMAGE_BYTES or time.monotonic() > deadline:
                                raise UrlError('image limit exceeded')
                            output.write(chunk)
                    if not size:
                        raise UrlError('image is empty')
                    return
            except requests.RequestException as exc:
                raise UrlError('image download transport failed') from exc
    raise UrlError('too many redirects')
