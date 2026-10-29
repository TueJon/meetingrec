from meetingrec import gui_model
from meetingrec.gui_model import Setup, call_candidates, refresh_for_today, short


def _window(wid, pid, title, binary="chrome", audio=False):
    return {"id": wid, "pid": pid, "title": title, "binary": binary, "audio": audio}


def _targets(windows=(), apps=(), mics=("mic0",), monitors=("DP-1",), session="x11"):
    return {
        "session": session,
        "windows": list(windows),
        "apps": list(apps),
        "mics": [{"name": m} for m in mics],
        "monitors": [{"name": m} for m in monitors],
    }


def test_argv_for_a_call_with_the_call_window(monkeypatch):
    monkeypatch.setattr(gui_model, "CLI", ["meetingrec"])
    setup = Setup(title="Standup", app_pid=42, screen="window", window_id=7, mic="usb", fast=True)
    assert setup.argv() == [
        "meetingrec", "full", "Standup", "--events", "--fast",
        "--app", "42", "--mic", "usb", "--screen", "window", "--window-id", "7",
    ]  # fmt: skip


def test_argv_room_ignores_the_call_app(monkeypatch):
    monkeypatch.setattr(gui_model, "CLI", ["meetingrec"])
    setup = Setup(room=True, app_pid=42, process=False, fast=True, screen="monitor", monitor="DP-1")
    assert setup.argv() == [
        "meetingrec", "record", "meeting", "--events", "--room", "--screen", "monitor", "--monitor", "DP-1",
    ]  # fmt: skip


def test_short_title():
    assert short("Microsoft Teams (PWA) - Chat | Someone | Microsoft Teams") == "Microsoft Teams (PWA)"
    assert short("Discord") == "Discord"


def test_call_candidates_one_row_per_process_unless_several_calls():
    windows = [
        _window(1, 100, "Microsoft Teams (PWA) - Chat"),
        _window(2, 100, "Outlook (PWA) - Mail"),
        _window(3, 200, "YouTube Music", binary="chrome", audio=True),
        _window(4, 300, "✳ Meetingrec UI wizard app", binary="gnome-terminal-"),
        _window(5, 300, "tj@host: ~", binary="gnome-terminal-"),
    ]
    playing = [
        {"pid": 400, "binary": "Discord", "app": "Discord"},
        {"pid": 200, "binary": "chrome", "app": "x"},
    ]
    suggested, others = call_candidates(windows, playing)
    assert [(e["pid"], e["title"]) for e in suggested] == [
        (200, "YouTube Music"),
        (100, "Microsoft Teams (PWA) - Chat"),
        (400, "Discord"),
    ]
    assert [e["window"] for e in others] == [4]  # "Meetingrec" is not a "meet" call


def test_refresh_finds_the_new_pid_of_the_same_call_window():
    setup = Setup(
        app_pid=1, app_binary="chrome", app_title="Teams", screen="window", window_id=9, window_title="Teams"
    )
    targets = _targets(windows=[_window(5, 100, "Other"), _window(6, 100, "Teams")])
    setup, notes = refresh_for_today(setup, targets)
    assert (setup.app_pid, setup.window_id, notes) == (100, 6, [])


def test_refresh_uses_a_windowless_app_and_falls_back_to_system_audio():
    setup, notes = refresh_for_today(
        Setup(app_pid=1, app_binary="Discord"), _targets(apps=[{"pid": 77, "binary": "Discord"}])
    )
    assert setup.app_pid == 77 and notes == []
    setup, notes = refresh_for_today(Setup(app_pid=1, app_binary="Discord"), _targets())
    assert (setup.app_pid, setup.app_binary) == (None, "")
    assert notes == ["The last call app is not open; recording all system audio instead."]


def test_refresh_turns_off_a_vanished_window_and_fixes_monitor_and_mic():
    setup = Setup(screen="window", window_id=9, window_title="Gone", mic="unplugged")
    setup, notes = refresh_for_today(setup, _targets())
    assert setup.screen == "off" and setup.mic == "" and len(notes) == 2
    setup, _ = refresh_for_today(Setup(screen="monitor", monitor="HDMI-1"), _targets())
    assert setup.monitor == "DP-1"


def test_db_fraction_clamps():
    assert gui_model.db_fraction(-700) == 0 and gui_model.db_fraction(0) == 1
    assert gui_model.db_fraction(-30) == 0.5
