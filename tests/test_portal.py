import stat

import pytest

from meetingrec.capture import portal


def test_request_path_strips_colon_and_dots():
    assert portal.request_path(":1.234", "mr_abc") == "/org/freedesktop/portal/desktop/request/1_234/mr_abc"


def test_window_never_persists_or_restores():
    opts = portal.select_sources_options("window", "t", cursor_modes=7, restore_token="old")
    assert opts["types"] == ("u", 2)
    assert opts["persist_mode"] == ("u", 0)
    assert "restore_token" not in opts
    assert opts["cursor_mode"] == ("u", 2)


def test_monitor_persists_and_restores():
    opts = portal.select_sources_options("monitor", "t", cursor_modes=7, restore_token="tok")
    assert opts["types"] == ("u", 1)
    assert opts["persist_mode"] == ("u", 2)
    assert opts["restore_token"] == ("s", "tok")
    assert opts["multiple"] == ("b", False)


def test_cursor_mode_omitted_when_embedded_unavailable():
    assert "cursor_mode" not in portal.select_sources_options(
        "monitor", "t", cursor_modes=1, restore_token=None
    )


def test_check_response():
    portal.check_response(0)
    with pytest.raises(portal.ScreenCastCancelled):
        portal.check_response(1)
    with pytest.raises(RuntimeError):
        portal.check_response(2)


def test_parse_start_results():
    results = {
        "streams": ("a(ua{sv})", [(57, {"size": ("(ii)", (1920, 1080)), "source_type": ("u", 1)})]),
        "restore_token": ("s", "new"),
    }
    assert portal.parse_start_results(results) == (57, 1920, 1080, "new")


def test_parse_start_results_without_size_or_token():
    assert portal.parse_start_results({"streams": ("a(ua{sv})", [(9, {})])}) == (9, None, None, None)


def test_parse_start_results_without_stream():
    with pytest.raises(RuntimeError):
        portal.parse_start_results({"streams": ("a(ua{sv})", [])})


def test_token_roundtrip_is_private(tmp_path):
    path = tmp_path / "sub" / "token"
    assert portal.read_token(path) is None
    portal.write_token(path, "abc")
    portal.write_token(path, "def")
    assert portal.read_token(path) == "def"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not path.with_name("token.tmp").exists()
