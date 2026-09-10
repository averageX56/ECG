import io
import urllib.error

import pytest

from scripts import download_qwen as module


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None, fail_after_first=False):
        super().__init__(data)
        self.status = status
        self.headers = headers or {'Content-Length': str(len(data))}
        self.fail_after_first = fail_after_first

    def read(self, size=-1):
        if self.fail_after_first and self.tell():
            raise TimeoutError('Read timed out')
        return super().read(size)


def test_timeout_resumes_from_bytes_written(tmp_path, monkeypatch):
    payload = b'a'*(2**20) + b'final data'
    requests, waits = [], []
    def open_request(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return Response(payload, fail_after_first=True)
        offset = 2**20
        return Response(payload[offset:], 206, {'Content-Range': f'bytes {offset}-{len(payload)-1}/{len(payload)}',
                                                'Content-Length': str(len(payload)-offset)})
    monkeypatch.setattr(module.urllib.request, 'urlopen', open_request)
    monkeypatch.setattr(module.time, 'sleep', waits.append)
    path = tmp_path/'weights.safetensors'
    module.download('https://example.test/weights', path)
    assert path.read_bytes() == payload
    assert requests[1].get_header('Range') == f'bytes={2**20}-'
    assert waits == [2] and not path.with_suffix('.safetensors.partial').exists()


def test_existing_partial_and_server_ignoring_range(tmp_path, monkeypatch):
    path = tmp_path/'weights'
    path.with_suffix('.partial').write_bytes(b'old prefix')
    seen = []
    def open_request(request, timeout):
        seen.append(request.get_header('Range'))
        return Response(b'complete new response')
    monkeypatch.setattr(module.urllib.request, 'urlopen', open_request)
    module.download('https://example.test/weights', path)
    assert seen == ['bytes=10-']
    assert path.read_bytes() == b'complete new response'
    module.download('https://example.test/weights', path)
    assert len(seen) == 1  # Completed files are reused.


def test_short_body_is_not_published(tmp_path, monkeypatch):
    path = tmp_path/'weights'
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: Response(b'abc', headers={'Content-Length': '10'}))
    with pytest.raises(RuntimeError, match='Partial file preserved'):
        module.download('https://example.test/weights', path, attempts=1)
    assert not path.exists() and path.with_suffix('.partial').read_bytes() == b'abc'


def test_invalid_range_cannot_corrupt_existing_partial(tmp_path, monkeypatch):
    path = tmp_path/'weights'
    path.with_suffix('.partial').write_bytes(b'abc')
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **k: Response(b'xyz', 206,
        {'Content-Range': 'bytes 0-2/6', 'Content-Length': '3'}))
    with pytest.raises(ValueError, match='Unexpected resume range'):
        module.download('https://example.test/weights', path)
    assert not path.exists() and path.with_suffix('.partial').read_bytes() == b'abc'


def test_http_error_policy_and_range_reset(tmp_path, monkeypatch):
    path = tmp_path/'weights'
    path.with_suffix('.partial').write_bytes(b'oversized')
    seen = []
    def open_request(request, timeout):
        seen.append(request.get_header('Range'))
        if len(seen) == 1:
            raise urllib.error.HTTPError(request.full_url, 416, 'Range invalid', {}, None)
        return Response(b'correct')
    monkeypatch.setattr(module.urllib.request, 'urlopen', open_request)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    module.download('https://example.test/weights', path)
    assert seen == ['bytes=9-', None] and path.read_bytes() == b'correct'

    def forbidden(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, None)
    monkeypatch.setattr(module.urllib.request, 'urlopen', forbidden)
    with pytest.raises(urllib.error.HTTPError):
        module.download('https://example.test/weights', tmp_path/'other')
