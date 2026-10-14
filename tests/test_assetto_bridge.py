"""Protocol compatibility checks for the optional native simulator bridge."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "assetto_runner", Path(__file__).parents[1] / "integrations/assetto_corsa/runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "reply, expected",
    [
        (b'{"CarName":"bmw_z4_gt3","active":true}', {"CarName": "bmw_z4_gt3", "active": True}),
        (
            b"{'sampling_freq':50,'screen_capture_enable':False}",
            {"sampling_freq": 50, "screen_capture_enable": False},
        ),
    ],
)
def test_management_accepts_both_author_wire_formats(monkeypatch, reply, expected):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sendall(self, message):
            assert message == b"get_config"

        def settimeout(self, timeout):
            pass

        def recv(self, maximum):
            return reply

    def connect(address, timeout):
        assert address == ("127.0.0.1", 2347)
        return Connection()

    monkeypatch.setattr(runner.socket, "create_connection", connect)
    assert runner.management("get_config") == expected


def test_management_does_not_execute_received_expressions(monkeypatch):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sendall(self, message):
            pass

        def settimeout(self, timeout):
            pass

        def recv(self, maximum):
            return b"__import__('builtins').print('not a state dictionary')"

    monkeypatch.setattr(runner.socket, "create_connection", lambda *args, **kwargs: Connection())
    with pytest.raises(ValueError):
        runner.management("get_static_info")
