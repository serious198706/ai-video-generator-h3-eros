"""HTTPS with validated, pinned DNS addresses. No ambient proxy or redirects for POST."""
import ipaddress
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit, urljoin
import urllib3

MAX_IMAGE_BYTES = 25 * 1024 * 1024


class UrlError(ValueError):
    pass


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
    allowed = [h.strip().lower() for h in hosts if h.strip()]
    host = host.lower()
    if allowed and not any(host == h or (h.startswith('.') and host.endswith(h) and host != h[1:]) for h in allowed):
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
    deadline = time.monotonic() + 90
    for _ in range(4):
        pool, response = request(url, hosts=hosts)
        try:
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                if not location:
                    raise UrlError('redirect without location')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise UrlError('image download rejected')
            if int(response.headers.get('Content-Length', '0')) > MAX_IMAGE_BYTES:
                raise UrlError('image too large')
            size = 0
            with destination.open('wb') as output:
                for chunk in response.stream(65536):
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES or time.monotonic() > deadline:
                        raise UrlError('image limit exceeded')
                    output.write(chunk)
            return
        finally:
            response.close()
            pool.close()
    raise UrlError('too many redirects')
