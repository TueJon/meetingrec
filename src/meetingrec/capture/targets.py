"""Everything a recording can target, for `meetingrec targets` and the GUI."""

from __future__ import annotations

import json
import os
import subprocess

from . import apps, x11


def parse_sources(sources: list[dict], default: str) -> list[dict]:
    return [
        {"name": s["name"], "description": s.get("description", ""), "default": s["name"] == default}
        for s in sources
        if not s.get("monitor_source") and not s["name"].startswith(apps.BUS_PREFIX)
    ]


def parse_wmctrl(output: str) -> list[dict]:
    """`wmctrl -lp` lines: id, desktop, pid, host, title. Sticky (-1) windows are panels."""
    windows = []
    for line in output.splitlines():
        fields = line.split(None, 4)
        if len(fields) < 4 or fields[1] == "-1":
            continue
        windows.append(
            {"id": int(fields[0], 16), "pid": int(fields[2]), "title": fields[4] if len(fields) > 4 else ""}
        )
    return windows


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def collect() -> dict:
    default = _run(["pactl", "get-default-source"]).strip()
    mics = parse_sources(json.loads(_run(["pactl", "-f", "json", "list", "sources"]) or "[]"), default)
    streams = apps.playing_streams()
    ancestors = {s.pid: apps.ancestry(s.pid) for s in streams if s.pid}
    is_x11 = x11.is_x11_session()
    windows = parse_wmctrl(_run(["wmctrl", "-lp"])) if is_x11 else []
    for w in windows:
        selector = apps.AppSelector(pid=w["pid"], binary=apps.process_binary(w["pid"]))
        w["binary"] = selector.binary
        w["audio"] = any(apps.matches(s, selector, ancestors) for s in streams)
    return {
        "session": "x11" if is_x11 else os.environ.get("XDG_SESSION_TYPE", "unknown"),
        "mics": mics,
        "apps": [{"binary": s.binary, "pid": s.pid, "app": s.app, "media": s.media} for s in streams],
        "windows": windows,
        "monitors": [vars(m) for m in x11.list_monitors()] if is_x11 else [],
    }
