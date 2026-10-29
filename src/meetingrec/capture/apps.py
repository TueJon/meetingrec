"""Record one application's audio (Teams, Discord, a browser) instead of the whole system mix.

A private null sink is the capture bus. A watcher thread links the output ports of every
playback stream that belongs to the chosen app into that bus, alongside the app's normal
link to the speakers, so the app stays audible and nothing else (music, notifications)
reaches the recording. Streams that appear later (a call joined after the start, a browser
re-creating its stream) are linked within `POLL_SECONDS`. The bus monitor runs on its own
clock, so the track is gap-free even while the app is silent.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

BUS_PREFIX = "meetingrec_app"
POLL_SECONDS = 0.5
_CHANNEL_FALLBACK = {"MONO": ("FL", "FR")}


@dataclass(frozen=True)
class Stream:
    node_id: int
    app: str
    binary: str
    pid: int | None
    media: str


@dataclass(frozen=True)
class Port:
    port_id: int
    node_id: int
    direction: str
    channel: str


@dataclass(frozen=True)
class AppSelector:
    """Matches streams of one process tree, or of one binary when the PID is unknown or sandboxed."""

    pid: int | None = None
    binary: str = ""

    @classmethod
    def parse(cls, value: str) -> AppSelector:
        if value.isdigit():
            pid = int(value)
            return cls(pid=pid, binary=process_binary(pid))
        return cls(binary=value)

    def label(self) -> str:
        return f"{self.binary or '?'} (pid {self.pid})" if self.pid else self.binary


# ---- pure helpers -----------------------------------------------------------


def parse_dump(objects: list[dict]) -> tuple[list[Stream], list[Port]]:
    """Playback streams and ports. Native PipeWire clients keep the process properties on the
    client object, PulseAudio clients (browsers, Electron) copy them onto the node."""
    clients = {
        o["id"]: (o.get("info") or {}).get("props") or {}
        for o in objects
        if o.get("type") == "PipeWire:Interface:Client"
    }
    streams, ports = [], []
    for obj in objects:
        props = (obj.get("info") or {}).get("props") or {}
        if obj.get("type") == "PipeWire:Interface:Node" and props.get("media.class") == "Stream/Output/Audio":
            props = {**clients.get(props.get("client.id"), {}), **props}
            pid = props.get("application.process.id")
            streams.append(
                Stream(
                    node_id=obj["id"],
                    app=props.get("application.name", ""),
                    binary=props.get("application.process.binary", ""),
                    pid=int(pid) if str(pid).isdigit() else None,
                    media=props.get("media.name", ""),
                )
            )
        elif obj.get("type") == "PipeWire:Interface:Port" and "node.id" in props:
            ports.append(
                Port(
                    port_id=obj["id"],
                    node_id=int(props["node.id"]),
                    direction=props.get("port.direction", ""),
                    channel=props.get("audio.channel", ""),
                )
            )
    return streams, ports


def matches(stream: Stream, selector: AppSelector, ancestors: dict[int, list[int]]) -> bool:
    """By process tree when both PIDs are known: two Chrome channels share the binary name."""
    if selector.pid and stream.pid:
        return selector.pid in ancestors.get(stream.pid, [stream.pid])
    return bool(selector.binary) and stream.binary == selector.binary


def link_pairs(stream_ports: list[Port], bus_ports: list[Port]) -> list[tuple[int, int]]:
    """(output port, bus input port) pairs, by channel; mono feeds both bus channels."""
    inputs = {p.channel: p.port_id for p in bus_ports if p.direction == "in"}
    pairs = []
    for port in stream_ports:
        if port.direction != "out":
            continue
        for channel in _CHANNEL_FALLBACK.get(port.channel, (port.channel,)):
            if channel in inputs:
                pairs.append((port.port_id, inputs[channel]))
    return pairs


# ---- system queries ---------------------------------------------------------


def process_binary(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def _parent(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[1])


def ancestry(pid: int) -> list[int]:
    chain = [pid]
    while (parent := _parent(chain[-1])) and parent > 1 and len(chain) < 64:
        chain.append(parent)
    return chain


def pw_dump() -> list[dict]:
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, check=True, timeout=10).stdout
    except FileNotFoundError:
        raise RuntimeError("pw-dump not found (install pipewire-bin)") from None
    return json.loads(out or "[]")


def playing_streams() -> list[Stream]:
    return parse_dump(pw_dump())[0]


def _pactl(*args: str) -> str:
    return subprocess.run(["pactl", *args], capture_output=True, text=True, check=True, timeout=10).stdout


def _unload_stale_buses() -> None:
    """Remove buses left behind by a crashed recorder; they are ours by name."""
    for line in _pactl("list", "short", "modules").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3 and fields[1] == "module-null-sink" and f"sink_name={BUS_PREFIX}" in fields[2]:
            subprocess.run(["pactl", "unload-module", fields[0]], capture_output=True, timeout=10)


# ---- the tap ----------------------------------------------------------------


class _Linker(threading.Thread):
    def __init__(self, bus: str, selector: AppSelector) -> None:
        super().__init__(daemon=True)
        self.bus = bus
        self.selector = selector
        self.stop_event = threading.Event()
        self.linked: set[tuple[int, int]] = set()
        self.apps: set[str] = set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            # PipeWire restarting or a node vanishing mid-dump; retry next tick.
            with suppress(RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
                self.sync()
            self.stop_event.wait(POLL_SECONDS)

    def sync(self) -> None:
        objects = pw_dump()
        streams, ports = parse_dump(objects)
        bus_node = next(
            (
                o["id"]
                for o in objects
                if ((o.get("info") or {}).get("props") or {}).get("node.name") == self.bus
            ),
            None,
        )
        if bus_node is None:
            return
        bus_ports = [p for p in ports if p.node_id == bus_node]
        ancestors = {s.pid: ancestry(s.pid) for s in streams if s.pid}
        for stream in streams:
            if not matches(stream, self.selector, ancestors):
                continue
            self.apps.add(stream.app or stream.binary)
            for pair in link_pairs([p for p in ports if p.node_id == stream.node_id], bus_ports):
                if pair in self.linked:
                    continue
                result = subprocess.run(
                    ["pw-link", str(pair[0]), str(pair[1])], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 or "exists" in result.stderr:
                    self.linked.add(pair)


@dataclass
class AppTap:
    source: str  # pulse source name to record: the bus monitor
    linker: _Linker


@contextmanager
def tap(selector: AppSelector) -> Iterator[AppTap]:
    _unload_stale_buses()
    bus = f"{BUS_PREFIX}_{os.getpid()}"
    description = f"meetingrec capture ({selector.label()})".replace(" ", "\\ ")
    module = _pactl(
        "load-module",
        "module-null-sink",
        f"sink_name={bus}",
        "channels=2",
        "rate=48000",
        f"sink_properties=node.description={description} priority.session=0 priority.driver=0",
    ).strip()
    linker = _Linker(bus, selector)
    try:
        linker.sync()
        linker.start()
        yield AppTap(source=f"{bus}.monitor", linker=linker)
    finally:
        linker.stop_event.set()
        if linker.is_alive():
            linker.join(timeout=5)
        subprocess.run(["pactl", "unload-module", module], capture_output=True, timeout=10)
