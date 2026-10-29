"""What the recorder window decides, without GTK: the saved setup, re-finding last time's call
app and window, ranking call candidates, and the CLI arguments. Stdlib only, so the window
(system Python) and the tests (venv) both import it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

CLI = json.loads(os.environ.get("MEETINGREC_CMD", '["meetingrec"]'))
STATE = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "meetingrec" / "gui.json"
METER_FLOOR_DB = -60.0


@dataclass
class Setup:
    title: str = "meeting"
    room: bool = False
    app_pid: int | None = None  # call audio: the window's process; None = whole system mix
    app_binary: str = ""
    app_title: str = ""
    mic: str = ""  # "" = system default
    screen: str = "off"  # off | window | monitor
    window_id: int | None = None
    window_title: str = ""
    monitor: str = ""
    process: bool = True
    fast: bool = False

    def argv(self) -> list[str]:
        argv = [*CLI, "full" if self.process else "record", self.title or "meeting", "--events"]
        if self.process and self.fast:
            argv.append("--fast")
        if self.room:
            argv.append("--room")
        elif self.app_pid:
            argv += ["--app", str(self.app_pid)]
        if self.mic:
            argv += ["--mic", self.mic]
        if self.screen != "off":
            argv += ["--screen", self.screen]
            if self.screen == "window" and self.window_id:
                argv += ["--window-id", str(self.window_id)]
            if self.screen == "monitor" and self.monitor:
                argv += ["--monitor", self.monitor]
        return argv

    def describe(self) -> str:
        if self.room:
            audio = "In the room: microphone only"
        elif self.app_pid:
            audio = f"Call audio from {short(self.app_title) or self.app_binary}"
        else:
            audio = "Microphone and all system audio"
        screen = {
            "off": "no screen",
            "window": "the call window"
            if self.window_title and self.window_title == self.app_title
            else f"window “{short(self.window_title)}”",
            "monitor": f"monitor {self.monitor}",
        }[self.screen]
        after = "transcript and summary afterwards" if self.process else "recording only"
        return f"{audio} · {screen} · {after}"


def short(title: str) -> str:
    """ "Microsoft Teams (PWA) - Chat | …" -> "Microsoft Teams (PWA)"."""
    return re.split(r" [-|·—] ", title, maxsplit=1)[0][:48]


def load_setup() -> Setup:
    try:
        return Setup(**{**asdict(Setup()), **json.loads(STATE.read_text())})
    except (OSError, ValueError, TypeError):
        return Setup()


def save_setup(setup: Setup) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(asdict(setup), indent=2) + "\n")


def fetch_targets() -> dict:
    out = subprocess.run([*CLI, "targets", "--json"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "meetingrec targets failed")
    return json.loads(out.stdout)


def refresh_for_today(setup: Setup, targets: dict) -> tuple[Setup, list[str]]:
    """Re-find the last call window and screen target: PIDs and window IDs change between runs."""
    notes = []
    windows = [w for w in targets["windows"] if w["pid"] != os.getpid()]
    if not setup.room and setup.app_binary:
        same_app = [w for w in windows if w["binary"] == setup.app_binary]
        best = next((w for w in same_app if w["title"] == setup.app_title), None)
        best = best or next((w for w in same_app if w["audio"]), None) or (same_app[0] if same_app else None)
        windowless = next((a for a in targets["apps"] if a["binary"] == setup.app_binary and a["pid"]), None)
        if best or windowless:
            setup.app_pid = (best or windowless)["pid"]
        else:
            setup.app_pid, setup.app_binary = None, ""
            notes.append("The last call app is not open; recording all system audio instead.")
    if setup.screen == "window":
        match = next((w for w in windows if w["id"] == setup.window_id), None)
        match = match or next((w for w in windows if w["title"] == setup.window_title), None)
        match = match or next((w for w in windows if setup.app_pid and w["pid"] == setup.app_pid), None)
        if match:
            setup.window_id, setup.window_title = match["id"], match["title"]
        elif targets["session"] == "x11":
            setup.screen = "off"
            notes.append("The last recorded window is gone; screen recording is off.")
    monitors = [m["name"] for m in targets["monitors"]]
    if setup.screen == "monitor" and monitors and setup.monitor not in monitors:
        setup.monitor = monitors[0]
    if setup.mic and setup.mic not in {m["name"] for m in targets["mics"]}:
        setup.mic = ""
        notes.append("The last microphone is not connected; using the default one.")
    return setup, notes


CALL_HINTS = re.compile(
    r"\b(teams|meet|discord|zoom|webex|slack|jitsi|whereby|skype|signal|whatsapp|telegram|huddle)\b",
    re.IGNORECASE,
)


def call_candidates(windows: list[dict], playing: list[dict]) -> tuple[list[dict], list[dict]]:
    """(suggested, other) call sources, at most one row per process unless several of its windows
    look like calls: a process's windows all share its audio. Suggested are call-looking windows,
    processes playing audio, and audio apps without a window."""
    by_pid: dict[int, list[dict]] = {}
    for w in windows:
        entry = {
            "pid": w["pid"],
            "binary": w["binary"],
            "title": w["title"],
            "audio": w["audio"],
            "window": w["id"],
        }
        by_pid.setdefault(w["pid"], []).append(entry)
    suggested, others = [], []
    for entries in by_pid.values():
        calls = [e for e in entries if CALL_HINTS.search(e["title"])]
        if calls:
            suggested += calls
        elif entries[0]["audio"]:
            suggested.append(entries[0])
        else:
            others.append(entries[0])
    suggested.sort(key=lambda e: (not e["audio"], not CALL_HINTS.search(e["title"])))
    suggested += [
        {
            "pid": a["pid"],
            "binary": a["binary"],
            "title": a["app"] or a["binary"],
            "audio": True,
            "window": None,
        }
        for a in playing
        if a["pid"] and a["pid"] not in by_pid
    ]
    return suggested, others


def entry_title(entry: dict) -> str:
    return f"♪ {entry['title']}" if entry["audio"] else entry["title"]


def db_fraction(db: float) -> float:
    return min(1.0, max(0.0, (db - METER_FLOOR_DB) / -METER_FLOOR_DB))
