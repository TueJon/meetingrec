"""xdg-desktop-portal ScreenCast client: ask the user (once) which window/monitor to share.

Returns a PipeWire remote fd + node id for GStreamer's `pipewiresrc`. Never used in
tests or `doctor` beyond a property read: Start pops a dialog on the user's screen.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jeepney import DBusAddress, MatchRule, message_bus, new_method_call
from jeepney.io.blocking import DBusConnection, open_dbus_connection

from ..config import data_dir

Source = Literal["window", "monitor"]

BUS_NAME = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"

SOURCE_TYPES: dict[str, int] = {"monitor": 1, "window": 2}
CURSOR_EMBEDDED = 2
PERSIST_UNTIL_REVOKED = 2

_SCREENCAST = DBusAddress(PORTAL_PATH, BUS_NAME, SCREENCAST_IFACE)
_PROPERTIES = "org.freedesktop.DBus.Properties"


class ScreenCastCancelled(Exception):
    """The user closed the picker dialog."""


@dataclass
class ScreenCast:
    fd: int
    node_id: int
    width: int | None
    height: int | None
    source_type: Source


# ---- pure helpers -----------------------------------------------------------


def request_path(unique_name: str, token: str) -> str:
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"


def select_sources_options(
    source: Source, token: str, *, cursor_modes: int, restore_token: str | None
) -> dict[str, tuple[str, Any]]:
    # Windows are never remembered: the picker shows every meeting (privacy default).
    opts: dict[str, tuple[str, Any]] = {
        "handle_token": ("s", token),
        "types": ("u", SOURCE_TYPES[source]),
        "multiple": ("b", False),
        "persist_mode": ("u", PERSIST_UNTIL_REVOKED if source == "monitor" else 0),
    }
    if cursor_modes & CURSOR_EMBEDDED:
        opts["cursor_mode"] = ("u", CURSOR_EMBEDDED)
    if source == "monitor" and restore_token:
        opts["restore_token"] = ("s", restore_token)
    return opts


def check_response(code: int) -> None:
    if code == 1:
        raise ScreenCastCancelled("screen sharing was cancelled")
    if code != 0:
        raise RuntimeError(f"ScreenCast portal request failed (response {code})")


def parse_start_results(
    results: dict[str, tuple[str, Any]],
) -> tuple[int, int | None, int | None, str | None]:
    """(node_id, width, height, restore_token) from the Start response."""
    streams = results.get("streams", ("", []))[1]
    if not streams:
        raise RuntimeError("ScreenCast portal returned no stream")
    node_id, props = streams[0]
    size = props.get("size", ("", None))[1]
    width, height = size if size else (None, None)
    token = results.get("restore_token", ("", None))[1]
    return node_id, width, height, token or None


def restore_token_path() -> Path:
    return data_dir() / "monitor-restore-token"


def read_token(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except FileNotFoundError:
        return None


def write_token(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token + "\n")
    os.replace(tmp, path)


# ---- D-Bus ------------------------------------------------------------------


def get_property(conn: DBusConnection, name: str) -> Any:
    msg = new_method_call(_SCREENCAST.with_interface(_PROPERTIES), "Get", "ss", (SCREENCAST_IFACE, name))
    return conn.send_and_get_reply(msg).body[0][1]


def screencast_version() -> int:
    """ScreenCast portal version, read-only (no dialog). Raises if the portal is unreachable."""
    with open_dbus_connection(bus="SESSION") as conn:
        return get_property(conn, "version")


def _call(conn: DBusConnection, method: str, signature: str, body: tuple) -> Any:
    reply = conn.send_and_get_reply(new_method_call(_SCREENCAST, method, signature, body))
    return reply.body


def _request(
    conn: DBusConnection, method: str, signature: str, body: tuple, token: str
) -> dict[str, tuple[str, Any]]:
    """Call a portal method and wait for its Request.Response (no timeout: the user may take minutes)."""
    path = request_path(conn.unique_name, token)
    rule = MatchRule(type="signal", interface=REQUEST_IFACE, member="Response", path=path)
    conn.send_and_get_reply(message_bus.AddMatch(rule))
    with conn.filter(rule) as matches:
        _call(conn, method, signature, body)
        while not matches:
            conn.recv_until_filtered(matches)
        code, results = matches.popleft().body
    conn.send_and_get_reply(message_bus.RemoveMatch(rule))
    check_response(code)
    return results


def _token() -> str:
    return "mr_" + secrets.token_hex(8)


@contextmanager
def screencast(source: Source, token_path: Path) -> Iterator[ScreenCast]:
    conn = open_dbus_connection(bus="SESSION", enable_fds=True)
    session_handle: str | None = None
    fd: int | None = None
    try:
        cursor_modes = get_property(conn, "AvailableCursorModes")
        token = _token()
        results = _request(
            conn,
            "CreateSession",
            "a{sv}",
            ({"handle_token": ("s", token), "session_handle_token": ("s", _token())},),
            token,
        )
        session_handle = results["session_handle"][1]

        token = _token()
        restore = read_token(token_path) if source == "monitor" else None
        opts = select_sources_options(source, token, cursor_modes=cursor_modes, restore_token=restore)
        _request(conn, "SelectSources", "oa{sv}", (session_handle, opts), token)

        token = _token()
        results = _request(
            conn, "Start", "osa{sv}", (session_handle, "", {"handle_token": ("s", token)}), token
        )
        node_id, width, height, new_token = parse_start_results(results)
        if source == "monitor" and new_token:
            write_token(token_path, new_token)

        (remote,) = _call(conn, "OpenPipeWireRemote", "oa{sv}", (session_handle, {}))
        fd = remote.to_raw_fd()
        yield ScreenCast(fd, node_id, width, height, source)
    finally:
        if fd is not None:
            os.close(fd)
        if session_handle is not None:
            close = new_method_call(DBusAddress(session_handle, BUS_NAME, SESSION_IFACE), "Close")
            # Best effort: the portal drops the session with the connection anyway.
            with suppress(Exception):
                conn.send_and_get_reply(close, timeout=2)
        conn.close()
