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


def test_argv_with_cloud_transcription_drops_fast(monkeypatch):
    monkeypatch.setattr(gui_model, "CLI", ["meetingrec"])
    setup = Setup(title="Standup", fast=True, asr="gemini")
    assert setup.argv() == ["meetingrec", "full", "Standup", "--events", "--asr", "gemini"]
    assert setup.describe().endswith("(Gemini cloud)")


def test_agent_prompt_lists_only_existing_files(tmp_path):
    (tmp_path / "summary.md").write_text("x")
    (tmp_path / "transcript.md").write_text("x")
    prompt = gui_model.agent_prompt("AI catchup", tmp_path)
    assert prompt.startswith(f"Meeting “AI catchup”, recorded in {tmp_path}:")
    assert f"- {tmp_path / 'summary.md'}: the minutes" in prompt
    assert f"- {tmp_path / 'transcript.md'}: the full transcript" in prompt
    assert "notes.md" not in prompt and "recording.mkv" not in prompt


def test_mic_name_resolves_the_default_and_unknown_sources():
    mics = [
        {"name": "usb", "description": "USB Mic", "default": False},
        {"name": "bt", "description": "Headset", "default": True},
    ]
    assert gui_model.mic_name("", mics) == "Headset"
    assert gui_model.mic_name("usb", mics) == "USB Mic"
    assert gui_model.mic_name("gone", mics) == "gone"
    assert gui_model.mic_name("", []) == "System default"


def test_argv_with_live_hints(monkeypatch):
    monkeypatch.setattr(gui_model, "CLI", ["meetingrec"])
    assert Setup(process=False, live=True, hints_in="/kb").argv()[-3:] == ["--live", "--hints-in", "/kb"]
    assert Setup(process=False, live=True).argv()[-1] == "--live"
    assert "--live" not in Setup(hints_in="/kb").argv()


def test_knowledge_bases_lists_existing_configured_folders(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "repo").mkdir()
    config = tmp_path / "config.toml"
    folders = ", ".join(f'"{tmp_path}/{name}"' for name in ("repo", "gone", "notes"))
    config.write_text(f'[live]\nworkdir = "{tmp_path}/notes"\nworkdirs = [{folders}]\n')
    assert gui_model.knowledge_bases(config) == [f"{tmp_path}/repo", f"{tmp_path}/notes"]
    assert gui_model.knowledge_bases(tmp_path / "missing.toml") == []


def test_a_hint_shows_its_age_and_a_short_source(tmp_path):
    assert [gui_model.age_text(s) for s in (2, 12.7, 59, 60, 185)] == [
        "now",
        "12 s ago",
        "59 s ago",
        "1 min ago",
        "3 min ago",
    ]
    assert gui_model.source_label("knowledge/a.md:12 (Scope)", "2026-11-16") == "a.md:12 · 2026-11-16"
    assert gui_model.source_label("somewhere", "unknown") == "somewhere"
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "a.md").write_text("x")
    assert gui_model.source_path("knowledge/a.md:12", str(tmp_path)) == tmp_path / "knowledge" / "a.md"
    assert gui_model.source_path("knowledge/b.md:1", str(tmp_path)) is None
    assert gui_model.source_path("knowledge/a.md:12", "") is None
