import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from h3_service.net.http import resolve, UrlError
from h3_service.media.images import prepare


def dns(ip):
    return [(2, 1, 6, '', (ip, 443))]


class NetworkTests(unittest.TestCase):
    def test_public(self):
        with patch('socket.getaddrinfo', return_value=dns('8.8.8.8')):
            self.assertEqual(resolve('https://cdn.example.com/x', ('.example.com',))[1], ['8.8.8.8'])

    def test_private_and_mapped_blocked(self):
        for ip in ['127.0.0.1', '169.254.169.254', '10.1.1.1', '::ffff:127.0.0.1']:
            with self.subTest(ip=ip), patch('socket.getaddrinfo', return_value=dns(ip)), self.assertRaises(UrlError):
                resolve('https://example.com/x')

    def test_callback_private_requires_allowlist(self):
        with patch('socket.getaddrinfo', return_value=dns('10.1.1.1')):
            with self.assertRaises(UrlError):
                resolve('https://callback.internal/x', webhook=True)
            resolve('https://callback.internal/x', ('callback.internal',), webhook=True)

    def test_all_dns_addresses_must_pass(self):
        with patch('socket.getaddrinfo', return_value=dns('8.8.8.8') + dns('127.0.0.1')), self.assertRaises(UrlError):
            resolve('https://example.com/x')

    def test_bad_urls(self):
        for url in ['http://example.com', 'https://user:pass@example.com', 'https://example.com:123', 'https://example.com/#x']:
            with self.subTest(url=url), self.assertRaises(UrlError):
                resolve(url)

    def test_last_image_does_not_choose_canvas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new('RGB', (300, 600), 'red').save(root / 'a.png')
            Image.new('RGB', (600, 300), 'blue').save(root / 'b.png')
            first, last, width, height = prepare(root / 'a.png', root / 'b.png', '540p', root)
            self.assertEqual((width, height), (544, 960))
            with Image.open(first) as a, Image.open(last) as b:
                self.assertEqual(a.size, b.size)
