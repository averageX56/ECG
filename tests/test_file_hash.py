import errno
import hashlib
import io

import numpy as np
import pytest

from ecg_project.data import catalog
from ecg_project.data.cache import shard


def test_hash_restarts_after_partial_read(monkeypatch, tmp_path, caplog):
    payload = b'a' * (1 << 20) + b'last chunk'
    streams = []

    class InterruptedRead(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError(121, 'Remote I/O error')
            return super().read(size)

    def open_file(*args):
        stream = InterruptedRead(payload) if not streams else io.BytesIO(payload)
        streams.append(stream)
        return stream

    sleeps = []
    monkeypatch.setattr(catalog, 'open', open_file, raising=False)
    monkeypatch.setattr(catalog.time, 'sleep', sleeps.append)
    path = tmp_path / 'record.npz'
    assert catalog.file_hash(path) == hashlib.sha256(payload).hexdigest()
    assert sleeps == [1]
    assert len(streams) == 2 and all(s.closed for s in streams)
    assert str(path) in caplog.text


@pytest.mark.parametrize('code', [getattr(errno, 'EREMOTEIO', 121), errno.EIO,
                                 errno.ETIMEDOUT, getattr(errno, 'ESTALE', 116),
                                 errno.EACCES, errno.ENOENT])
def test_hash_failure_is_bounded(monkeypatch, tmp_path, code):
    calls, sleeps = [], []

    def fail(*args):
        calls.append(1)
        raise OSError(code, 'storage failure')

    monkeypatch.setattr(catalog, 'open', fail, raising=False)
    monkeypatch.setattr(catalog.time, 'sleep', sleeps.append)
    path = tmp_path / 'record.npz'
    with pytest.raises(OSError) as caught:
        catalog.file_hash(path)
    assert caught.value.errno == code
    if code in (errno.EACCES, errno.ENOENT):
        assert len(calls) == 1 and sleeps == []
    else:
        assert len(calls) == 5 and sleeps == [1, 2, 4, 8]
        assert caught.value.filename == str(path)


def test_shard_resume_after_io_failure_still_checks_integrity(monkeypatch, tmp_path):
    path = shard(tmp_path, 'record', {}, lambda: {'x': np.arange(4)})
    real_open = open
    calls = []

    def flaky_open(*args):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(121, 'Remote I/O error')
        return real_open(*args)

    monkeypatch.setattr(catalog, 'open', flaky_open, raising=False)
    monkeypatch.setattr(catalog.time, 'sleep', lambda _: None)
    def compute():
        pytest.fail('Completed shard must not be recomputed')
    assert shard(tmp_path, 'record', {}, compute) == path
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Stale/modified'):
        shard(tmp_path, 'record', {}, compute)
