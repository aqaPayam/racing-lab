"""Atomic status writes survive temporary reader locks without hiding failures."""

# ruff: noqa: E402
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")

from racing_lab.rl import training


def test_transient_replacement_denial_preserves_old_status_until_success(tmp_path, monkeypatch):
    target = tmp_path / "queue.json"
    training.write_json(target, {"status": "old"})
    original_replace = Path.replace
    delays = []

    def replace(source, destination):
        if len(delays) < 2:
            assert json.loads(target.read_text()) == {"status": "old"}
            raise PermissionError("temporary reader lock")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(training.time, "sleep", delays.append)
    training.write_json(target, {"status": "completed"})
    assert json.loads(target.read_text()) == {"status": "completed"}
    assert len(delays) == 2
    assert not target.with_suffix(".json.tmp").exists()


def test_permanent_permission_denial_propagates_and_retains_evidence(tmp_path, monkeypatch):
    target = tmp_path / "queue.json"
    training.write_json(target, {"status": "old"})
    delays = []

    def denied(source, destination):
        raise PermissionError("permanent denial")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(training.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="permanent denial"):
        training.write_json(target, {"status": "new"})
    assert len(delays) < 40  # A permanent problem must not stall the queue forever.
    assert json.loads(target.read_text()) == {"status": "old"}
    assert json.loads(target.with_suffix(".json.tmp").read_text()) == {"status": "new"}


def test_other_io_errors_are_not_retried(tmp_path, monkeypatch):
    def missing(source, destination):
        raise FileNotFoundError("temporary file missing")

    def unexpected_sleep(seconds):
        pytest.fail("A non-permission error must propagate immediately")

    monkeypatch.setattr(Path, "replace", missing)
    monkeypatch.setattr(training.time, "sleep", unexpected_sleep)
    with pytest.raises(FileNotFoundError, match="temporary file missing"):
        training.write_json(tmp_path / "queue.json", {})


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows file-sharing regression")
def test_windows_reader_lock_is_retried_until_released(tmp_path, monkeypatch):
    target = tmp_path / "queue.json"
    training.write_json(target, {"status": "old"})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    # Permit reads/writes but deny deletion/replacement while this reader is open.
    handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    delays = []

    def release_on_retry(seconds):
        nonlocal handle
        delays.append(seconds)
        assert json.loads(target.read_text()) == {"status": "old"}
        assert kernel.CloseHandle(handle)
        handle = None

    try:
        monkeypatch.setattr(training.time, "sleep", release_on_retry)
        training.write_json(target, {"status": "completed"})
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
    assert len(delays) == 1  # The real OS lock, rather than a mocked exception, caused a retry.
    assert json.loads(target.read_text()) == {"status": "completed"}
