import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import requests
from h3_service.net.http import download, UrlError


def response(status=200, chunks=(b'image',), location=None):
    result = MagicMock(status_code=status, is_redirect=bool(location))
    result.__enter__.return_value = result
    result.headers = {'Location': location} if location else {}
    result.iter_content.return_value = iter(chunks)
    return result


class DownloadTransportTests(unittest.TestCase):
    def test_wan_hostname_session_timeout_and_user_agent(self):
        with tempfile.TemporaryDirectory() as temp, patch('requests.Session') as session, \
                patch('h3_service.net.http.resolve') as guard:
            client = session.return_value.__enter__.return_value
            client.get.return_value = response(chunks=(b'a', b'', b'bc'))
            file = Path(temp) / 'first'
            download('https://example.com/image', file)
            self.assertEqual(file.read_bytes(), b'abc')
            self.assertEqual(client.get.call_args.args[0], 'https://example.com/image')
            self.assertEqual(client.get.call_args.kwargs['timeout'], 30)
            self.assertFalse(client.get.call_args.kwargs['allow_redirects'])
            self.assertEqual(client.get.call_args.kwargs['headers'], {'User-Agent': 'wan22-server/1'})
            guard.assert_called_once()

    def test_redirect_is_checked_and_responses_closed(self):
        first, second = response(status=302, location='/next'), response()
        with tempfile.TemporaryDirectory() as temp, patch('requests.Session') as session, \
                patch('h3_service.net.http.resolve') as guard:
            client = session.return_value.__enter__.return_value
            client.get.side_effect = [first, second]
            download('https://example.com/start', Path(temp) / 'image')
            self.assertEqual([c.args[0] for c in guard.call_args_list],
                             ['https://example.com/start', 'https://example.com/next'])
            first.__exit__.assert_called_once()
            second.__exit__.assert_called_once()

    def test_http_status_and_transport_causes_preserved(self):
        with tempfile.TemporaryDirectory() as temp, patch('requests.Session') as session, \
                patch('h3_service.net.http.resolve'):
            client = session.return_value.__enter__.return_value
            client.get.return_value = response(status=403)
            with self.assertRaises(UrlError) as error:
                download('https://example.com/image', Path(temp) / 'image')
            self.assertEqual(error.exception.http_status, 403)
            client.get.side_effect = requests.exceptions.SSLError('private URL must not enter logs')
            with self.assertRaises(UrlError) as error:
                download('https://example.com/image', Path(temp) / 'image')
            from h3_service.tasks.worker import input_failure
            summary = input_failure(error.exception)
            self.assertIn('SSLError', summary)
            self.assertNotIn('private URL', summary)
