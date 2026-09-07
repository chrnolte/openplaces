"""Guards for the download staging path and the archive-type probe.

`download` infers a filename when handed a directory, stages the bytes
under a temporary name, and moves them into place. `unzip` picks an
extractor from the file. Each test below covers a case where one of
those steps produced a wrong path or a misleading error.
"""

import bz2
import threading
from pathlib import Path

import pytest

from openplaces import io as op_io
from openplaces.io import _needs_7z, download, unzip


class _FakeResponse:
    def __init__(self, headers, body=b'payload', delay=None):
        self.headers = headers
        self._body = body
        self._delay = delay

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        if self._delay is not None:
            self._delay.wait(timeout=5)
        yield self._body


def _serve(monkeypatch, headers, body=b'payload', delay=None):
    """Answer every request with one canned response."""

    def _get(url, **kwargs):
        return _FakeResponse(headers, body=body, delay=delay)

    monkeypatch.setattr(op_io.requests, 'head', _get)
    monkeypatch.setattr(op_io.requests, 'get', _get)


def test_content_disposition_without_a_filename_falls_back_to_the_url(
    tmp_path, monkeypatch
):
    # `attachment` with no filename= used to skip the URL fallback, so
    # the destination stayed the directory itself.
    _serve(monkeypatch, {'content-disposition': 'attachment'})
    out = download('https://example.invalid/data/parcels.zip', tmp_path)

    assert out == tmp_path / 'parcels.zip'
    assert out.is_file()
    assert out.read_bytes() == b'payload'
    assert tmp_path.is_dir()


def test_content_disposition_filename_is_used_when_present(tmp_path, monkeypatch):
    _serve(
        monkeypatch,
        {'content-disposition': 'attachment; filename="named.zip"'},
    )
    out = download('https://example.invalid/data/ignored.zip', tmp_path)
    assert out == tmp_path / 'named.zip'


def test_a_url_with_no_filename_still_writes_inside_the_directory(
    tmp_path, monkeypatch
):
    _serve(monkeypatch, {'content-type': 'application/zip'})
    out = download('https://example.invalid/', tmp_path)

    assert out.parent == tmp_path
    assert out.is_file()
    assert tmp_path.is_dir()


def test_concurrent_downloads_of_one_basename_do_not_mix(tmp_path, monkeypatch):
    # Both jobs fetch `parcels.zip`; a staging file keyed on the
    # basename alone put them in the same system-temp path.
    release = threading.Event()
    bodies = {'a': b'A' * 64, 'b': b'B' * 64}
    started = threading.Barrier(2, timeout=5)

    def _get(url, **kwargs):
        key = url.rsplit('/', 2)[-2]
        return _FakeResponse(
            {'content-disposition': 'attachment; filename="parcels.zip"'},
            body=bodies[key],
            delay=_Gate(started, release),
        )

    monkeypatch.setattr(op_io.requests, 'head', _get)
    monkeypatch.setattr(op_io.requests, 'get', _get)

    results = {}

    def _run(key):
        target = tmp_path / key
        target.mkdir()
        results[key] = download(f'https://example.invalid/{key}/parcels.zip', target)

    threads = [threading.Thread(target=_run, args=(k,)) for k in bodies]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(timeout=10)

    for key, body in bodies.items():
        assert results[key].read_bytes() == body


class _Gate:
    """Holds both downloads open until each has opened its staging file."""

    def __init__(self, barrier, release):
        self._barrier = barrier
        self._release = release

    def wait(self, timeout=None):
        self._barrier.wait()
        self._release.wait(timeout)


def test_needs_7z_is_true_for_a_container_zipfile_cannot_open(tmp_path):
    # A 7z, rar or bare gz archive: the probe used to raise BadZipFile
    # here, so the 7z fallback was unreachable.
    path = tmp_path / 'parcels.7z'
    path.write_bytes(b'7z\xbc\xaf\x27\x1c' + b'\x00' * 32)
    assert _needs_7z(path) is True


def test_needs_7z_is_false_for_a_plain_deflate_zip(tmp_path):
    from zipfile import ZIP_DEFLATED, ZipFile

    path = tmp_path / 'parcels.zip'
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('a.txt', 'x')
    assert _needs_7z(path) is False


def test_uppercase_bz2_is_decompressed(tmp_path):
    path = tmp_path / 'PARCELS.CSV.BZ2'
    path.write_bytes(bz2.compress(b'id,value\n1,2\n'))

    out_dir = unzip(path, tmp_path / 'out')
    assert (out_dir / 'PARCELS.CSV').read_bytes() == b'id,value\n1,2\n'


def test_uppercase_tar_bz2_is_extracted(tmp_path):
    import io
    import tarfile

    payload = b'id,value\n1,2\n'
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:bz2') as tar:
        info = tarfile.TarInfo('PARCELS.CSV')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    path = tmp_path / 'PARCELS.TAR.BZ2'
    path.write_bytes(buffer.getvalue())

    out_dir = unzip(path, tmp_path / 'out')
    assert (out_dir / 'PARCELS.CSV').read_bytes() == payload


def test_a_non_zip_archive_no_longer_reports_not_a_zip_file(tmp_path, monkeypatch):
    # With 7z absent the message says the archive cannot be extracted,
    # rather than the misleading 'File is not a zip file'.
    monkeypatch.setattr(op_io, '_find_7z', lambda: None)
    path = tmp_path / 'parcels.rar'
    path.write_bytes(b'Rar!\x1a\x07\x00' + b'\x00' * 32)

    with pytest.raises(RuntimeError, match='cannot be extracted'):
        unzip(path, tmp_path / 'out')


def test_a_finished_download_leaves_no_staging_directory(tmp_path, monkeypatch):
    _serve(monkeypatch, {'content-disposition': 'attachment; filename="a.zip"'})
    staged = []
    real_mkdtemp = op_io.tempfile.mkdtemp

    def _mkdtemp(*args, **kwargs):
        made = real_mkdtemp(*args, **kwargs)
        staged.append(Path(made))
        return made

    monkeypatch.setattr(op_io.tempfile, 'mkdtemp', _mkdtemp)
    download('https://example.invalid/a.zip', tmp_path)

    assert staged
    assert not any(directory.exists() for directory in staged)
