"""X11 screen capture: pick a window or monitor with xdotool/xrandr, record it with `ximagesrc`.

The xdg-desktop-portal picker never appears on GNOME/X11, so X11 sessions skip the portal.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..config import data_dir
from .portal import ScreenCastCancelled

Source = Literal["window", "monitor"]

XIMAGESRC_OPTIONS = ["use-damage=false", "show-pointer=true"]
_MONITOR_LINE = re.compile(r"^\s*\d+:\s+\S+\s+(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)\s+(\S+)\s*$")


@dataclass
class Monitor:
    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class X11Capture:
    source: list[str]
    width: int
    height: int
    source_type: Source


# ---- pure helpers -----------------------------------------------------------


def is_x11_session() -> bool:
    return os.environ.get("XDG_SESSION_TYPE") == "x11"


def monitor_choice_path() -> Path:
    return data_dir() / "x11-monitor"


def parse_monitors(output: str) -> list[Monitor]:
    monitors = []
    for line in output.splitlines():
        match = _MONITOR_LINE.match(line)
        if match:
            width, height, x, y, name = match.groups()
            monitors.append(Monitor(name, int(x), int(y), int(width), int(height)))
    return monitors


def parse_geometry(output: str) -> tuple[int, int]:
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    return int(values["WIDTH"]), int(values["HEIGHT"])


def window_source(window_id: int) -> list[str]:
    return ["ximagesrc", f"xid={window_id}", *XIMAGESRC_OPTIONS]


def monitor_source(monitor: Monitor) -> list[str]:
    return [
        "ximagesrc",
        f"startx={monitor.x}",
        f"starty={monitor.y}",
        f"endx={monitor.x + monitor.width - 1}",
        f"endy={monitor.y + monitor.height - 1}",
        *XIMAGESRC_OPTIONS,
    ]


def pick_monitor(monitors: list[Monitor], remembered: str | None, answer: str) -> Monitor:
    """The remembered monitor if still active, else the one the user typed (a 0-based index)."""
    if not monitors:
        raise RuntimeError("xrandr reports no active monitors")
    if len(monitors) == 1:
        return monitors[0]
    for monitor in monitors:
        if monitor.name == remembered:
            return monitor
    try:
        return monitors[int(answer)]
    except (ValueError, IndexError):
        raise RuntimeError(f"invalid monitor {answer!r}") from None


# ---- subprocess calls -------------------------------------------------------


def _run(argv: list[str], timeout: float | None = 60) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout).stdout
    except FileNotFoundError:
        raise RuntimeError(f"{argv[0]} not found (install xdotool and x11-xserver-utils)") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{' '.join(argv)} failed: {exc.stderr.strip()}") from exc


def select_window(window_id: int | None = None) -> X11Capture:
    if window_id is None:
        print("Click the window to record ...", flush=True)
        try:
            window_id = int(_run(["xdotool", "selectwindow"], timeout=None).strip())
        except RuntimeError as exc:
            raise ScreenCastCancelled(str(exc)) from None
    width, height = parse_geometry(_run(["xdotool", "getwindowgeometry", "--shell", str(window_id)]))
    return X11Capture(window_source(window_id), width, height, "window")


def list_monitors() -> list[Monitor]:
    return parse_monitors(_run(["xrandr", "--listactivemonitors"]))


def select_monitor(name: str | None = None) -> X11Capture:
    monitors = list_monitors()
    if name:
        monitor = next((m for m in monitors if m.name == name), None)
        if monitor is None:
            raise RuntimeError(f"monitor {name!r} is not active ({', '.join(m.name for m in monitors)})")
        return X11Capture(monitor_source(monitor), monitor.width, monitor.height, "monitor")
    path = monitor_choice_path()
    remembered = path.read_text().strip() if path.exists() else None
    answer = ""
    if len(monitors) > 1 and remembered not in {m.name for m in monitors}:
        for i, m in enumerate(monitors):
            print(f"  {i}: {m.name} {m.width}x{m.height}+{m.x}+{m.y}")
        if not sys.stdin.isatty():
            raise RuntimeError("several monitors and no terminal to ask: pass --monitor NAME")
        answer = input("Monitor to record: ").strip()
    monitor = pick_monitor(monitors, remembered, answer)
    if remembered == monitor.name:
        print(f"Using remembered monitor {monitor.name} (delete {path} to choose again).")
    elif len(monitors) > 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(monitor.name + "\n")
        print(f"Remembered monitor {monitor.name} (delete {path} to choose again).")
    return X11Capture(monitor_source(monitor), monitor.width, monitor.height, "monitor")


def capture(source: Source, *, window_id: int | None = None, monitor: str | None = None) -> X11Capture:
    return select_window(window_id) if source == "window" else select_monitor(monitor)
