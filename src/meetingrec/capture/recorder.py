"""Record mic + system (or one app's) audio (+ optional screen) into one Matroska file with GStreamer.

One `gst-launch-1.0` process owns every track, so all of them share one clock and
start at the same instant: that is what keeps audio and video in sync. `audiorate`
turns capture dropouts into silence instead of holes, so a stalled mic cannot shift
the timeline, and `level` reports each track's loudness for live meters and the
dropout check.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__
from ..config import Config
from ..session import Session, TrackMeta, VideoMeta
from . import apps, x11
from .portal import ScreenCastCancelled, restore_token_path, screencast

AUDIO_ROLES = ("mic", "system", "room")
FINALIZE_TIMEOUT = 30
# pulsesrc ring buffer and the queue behind it: a loaded laptop stalls the internal mic
# for up to ~1.4 s, which the default 200 ms buffer turns into lost audio.
PULSE_BUFFER_US = 2_000_000
PULSE_LATENCY_US = 20_000
QUEUE_NS = 10_000_000_000
LEVEL_INTERVAL_NS = 500_000_000
# A real microphone never delivers digital zero; a block this quiet is a dropout
# that audiorate filled (or a hardware-muted mic).
DROPOUT_DB = -150.0
DROPOUT_WARN_SECONDS = 2.0
DROPOUT_WINDOW_SECONDS = 10.0
WARN_EVERY_SECONDS = 30.0
MIN_FILE_BYTES = 1024
KEYFRAME_SECONDS = 10
VIDEO_BITRATE_KBPS = 400


# ---- source resolution --------------------------------------------------------


def _pactl(*args: str) -> str:
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True, check=True, timeout=10).stdout
    except FileNotFoundError:
        raise RuntimeError("pactl not found (install pulseaudio-utils / pipewire-pulse)") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pactl {' '.join(args)} failed: {exc.stderr.strip()}") from exc


def list_sources() -> list[str]:
    return [line.split("\t")[1] for line in _pactl("list", "short", "sources").splitlines() if "\t" in line]


def resolve_source(
    requested: str | None, configured: str, default: str, available: list[str], label: str
) -> str:
    name = requested or configured or default
    if name not in available:
        listing = "\n  ".join(available) or "(none)"
        raise RuntimeError(f"{label} source {name!r} not found. Available sources:\n  {listing}")
    return name


def resolve_mic(cfg: Config, mic: str | None) -> str:
    return resolve_source(mic, cfg.capture.mic, _pactl("get-default-source").strip(), list_sources(), "mic")


def resolve_system(cfg: Config, system: str | None) -> str:
    default_monitor = _pactl("get-default-sink").strip() + ".monitor"
    return resolve_source(system, cfg.capture.system, default_monitor, list_sources(), "system")


def resolve_sources(cfg: Config, mic: str | None, system: str | None) -> tuple[str, str]:
    return resolve_mic(cfg, mic), resolve_system(cfg, system)


# ---- pipeline ---------------------------------------------------------------


@dataclass
class VideoSpec:
    source: list[str]
    width: int
    height: int
    fps: int
    encoder: str


def _even(n: int) -> int:
    return max(2, n - n % 2)


def nvenc_works() -> bool:
    try:
        probe = subprocess.run(
            ["gst-launch-1.0", "-q", "videotestsrc", "num-buffers=5", "!", "nvh264enc", "!", "fakesink"],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def resolve_encoder(setting: str) -> str:
    if setting == "auto":
        return "nvh264enc" if nvenc_works() else "x264enc"
    return setting


def encoder_args(encoder: str, fps: int) -> list[str]:
    gop = str(fps * KEYFRAME_SECONDS)
    if encoder == "nvh264enc":
        return [encoder, f"bitrate={VIDEO_BITRATE_KBPS}", f"gop-size={gop}"]
    if encoder == "x264enc":
        return [
            encoder,
            f"bitrate={VIDEO_BITRATE_KBPS}",
            "speed-preset=veryfast",
            "tune=stillimage",
            "rc-lookahead=0",
            "bframes=0",
            "threads=1",
            f"key-int-max={gop}",
        ]
    raise ValueError(f"unsupported video encoder {encoder!r} (use auto, nvh264enc or x264enc)")


def _audio_branch(source: str, role: str) -> list[str]:
    return [
        "pulsesrc",
        f"device={source}",
        "do-timestamp=true",
        f"buffer-time={PULSE_BUFFER_US}",
        f"latency-time={PULSE_LATENCY_US}",
        "!",
        "queue",
        f"max-size-time={QUEUE_NS}",
        "max-size-buffers=0",
        "max-size-bytes=0",
        "!",
        "audioconvert",
        "!",
        "audioresample",
        "!",
        "audio/x-raw,rate=48000,channels=1",
        "!",
        "audiorate",
        "!",
        "level",
        f"name=level_{role}",
        f"interval={LEVEL_INTERVAL_NS}",
        "!",
        "flacenc",
        "!",
        "taginject",
        f"tags=title={role}",
        "!",
        "mux.",
    ]


def _video_branch(video: VideoSpec) -> list[str]:
    w, h = _even(video.width), _even(video.height)
    return [
        *video.source,
        "!",
        "queue",
        "!",
        "videoconvert",
        "!",
        "videorate",
        "!",
        f"video/x-raw,framerate={video.fps}/1",
        "!",
        "videoscale",
        "add-borders=true",
        "!",
        f"video/x-raw,width={w},height={h},pixel-aspect-ratio=1/1",
        "!",
        *encoder_args(video.encoder, video.fps),
        "!",
        "h264parse",
        "!",
        "mux.",
    ]


def build_pipeline(
    *,
    audio: list[tuple[str, str]],
    output: Path,
    video: VideoSpec | None = None,
) -> list[str]:
    """`audio` is (role, pulse source) per track, in stream order. `-m` prints the level messages."""
    argv = ["gst-launch-1.0", "-m", "-e", "matroskamux", "name=mux", "!", "filesink", f"location={output}"]
    for role, source in audio:
        argv += _audio_branch(source, role)
    if video:
        argv += _video_branch(video)
    return argv


# ---- stream mapping -----------------------------------------------------------


def map_audio_streams(probe: dict, roles: list[str]) -> dict[str, int]:
    """Role -> 0-based index among the audio streams, from ffprobe's title tags."""
    audio = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    # Matroska tags come back upper-cased ("TITLE") from ffprobe.
    mapping = {
        {k.lower(): v for k, v in s.get("tags", {}).items()}.get("title"): i for i, s in enumerate(audio)
    }
    missing = [r for r in roles if r not in mapping]
    if missing:
        raise RuntimeError(f"recording lacks audio track(s): {', '.join(missing)}")
    return {r: mapping[r] for r in roles}


def probe_file(path: Path) -> dict:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:stream_tags=title",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(out)


# ---- live levels and capture health -------------------------------------------

Emit = Callable[[dict], None]

_LEVEL_LINE = re.compile(
    r'element "level_(\w+)" \(element\): level, '
    r".*?rms=\(GValueArray\)< ([^>]+) >, peak=\(GValueArray\)< ([^>]+) >"
)


def parse_level(line: str) -> tuple[str, float, float] | None:
    """(role, rms dB, peak dB) from one `gst-launch -m` level message line."""
    match = _LEVEL_LINE.search(line)
    if not match:
        return None
    role, rms, peak = match.groups()
    return role, float(rms.split(",")[0]), float(peak.split(",")[0])


@dataclass
class Health:
    """Counts dropout seconds on the microphone track and warns while it happens."""

    emit: Emit
    mic_role: str
    dropout_seconds: float = 0.0
    _recent: list[float] = field(default_factory=list)
    _last_warning: float = -math.inf

    def level(self, role: str, rms: float, peak: float, now: float) -> None:
        self.emit({"event": "level", "role": role, "rms": round(rms, 1), "peak": round(peak, 1)})
        if role != self.mic_role or rms > DROPOUT_DB:
            return
        block = LEVEL_INTERVAL_NS / 1e9
        self.dropout_seconds += block
        self._recent = [t for t in self._recent if now - t < DROPOUT_WINDOW_SECONDS] + [now]
        if (
            len(self._recent) * block >= DROPOUT_WARN_SECONDS
            and now - self._last_warning >= WARN_EVERY_SECONDS
        ):
            self._last_warning = now
            self.emit(
                {
                    "event": "warning",
                    "kind": "mic-dropout",
                    "message": "The microphone is delivering no audio (muted, or dropping under load).",
                }
            )


def print_event(event: dict) -> None:
    if event["event"] == "warning":
        print(f"warning: {event['message']}", file=sys.stderr, flush=True)


def json_event(event: dict) -> None:
    print(json.dumps(event), flush=True)


# ---- process control ----------------------------------------------------------


class _Child:
    """gst-launch in its own session, so the terminal's Ctrl+C reaches only us and
    the child gets exactly one SIGINT (which `-e` turns into EOS)."""

    def __init__(self, argv: list[str], fds: tuple[int, ...], health: Health, log: Path) -> None:
        self.proc = subprocess.Popen(
            argv, pass_fds=fds, start_new_session=True, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._reader = threading.Thread(target=self._read, args=(health, log), daemon=True)
        self._reader.start()
        self._stopped = False
        self._previous = {
            sig: signal.signal(sig, lambda *_: self.stop()) for sig in (signal.SIGTERM, signal.SIGHUP)
        }
        # A parent that started us in the background (a GUI, `cmd &` in a script) may have left
        # SIGINT ignored, and Ctrl+C or the GUI's Stop would then never reach us.
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def _read(self, health: Health, log: Path) -> None:
        with log.open("w") as fh:
            for line in self.proc.stdout:
                level = parse_level(line)
                if level:
                    health.level(*level, now=time.monotonic())
                else:
                    fh.write(line)

    def stop(self) -> None:
        if not self._stopped and self.proc.poll() is None:
            self._stopped = True
            self.proc.send_signal(signal.SIGINT)

    def run(self) -> int:
        try:
            self.proc.wait()
        except KeyboardInterrupt:
            self.stop()
        try:
            return self._finalize()
        finally:
            # Processing follows; there SIGTERM must end the process again, not a finished child.
            for sig, handler in self._previous.items():
                signal.signal(sig, handler)

    def _finalize(self) -> int:
        if self._stopped:
            print("Stopping, finalizing the file ...", flush=True)
        while True:
            try:
                code = self.proc.wait(timeout=FINALIZE_TIMEOUT)
                self._reader.join(timeout=5)
                return code
            except subprocess.TimeoutExpired:
                print(
                    f"warning: recorder did not finish within {FINALIZE_TIMEOUT} s, killing it",
                    file=sys.stderr,
                )
                self.proc.kill()
            except KeyboardInterrupt:
                continue  # already stopping; a second Ctrl+C changes nothing


# ---- entry point ------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def pipewire_source(fd: int, node_id: int, fps: int) -> list[str]:
    return [
        "pipewiresrc",
        f"fd={fd}",
        f"path={node_id}",
        "do-timestamp=true",
        f"keepalive-time={1000 // fps}",
    ]


@contextmanager
def _screen_source(
    screen: str, fps: int, window_id: int | None, monitor: str | None
) -> Iterator[tuple[list[str], int, int, str, tuple[int, ...]]]:
    """(gst source elements, width, height, source type, fds to pass to gst-launch)."""
    if x11.is_x11_session():
        picked = x11.capture(screen, window_id=window_id, monitor=monitor)
        yield picked.source, picked.width, picked.height, picked.source_type, ()
        return
    with screencast(screen, restore_token_path()) as sc:
        yield (
            pipewire_source(sc.fd, sc.node_id, fps),
            sc.width or 1920,
            sc.height or 1080,
            sc.source_type,
            (sc.fd,),
        )


def record(
    session: Session,
    cfg: Config,
    *,
    screen: str | None,
    mic: str | None,
    system: str | None,
    app: str | None = None,
    room: bool = False,
    window_id: int | None = None,
    monitor: str | None = None,
    emit: Emit = print_event,
) -> None:
    """Room mode records the microphone alone; `app` records one application instead of the system mix."""
    if room and (app or system):
        raise RuntimeError("room mode records the microphone only; drop --app/--system")
    if app and system:
        raise RuntimeError("--app and --system are alternatives")
    output = session.path("recording.mkv" if screen else "recording.mka")
    session.meta.recorder_version = __version__
    session.meta.mode = "room" if room else "call"
    mic_src = resolve_mic(cfg, mic)
    if _pactl("get-source-mute", mic_src).strip().endswith("yes"):
        emit({"event": "warning", "kind": "mic-muted", "message": "The microphone is muted."})

    with ExitStack() as stack:
        audio = [("room" if room else "mic", mic_src)]
        if app:
            selector = apps.AppSelector.parse(app)
            bus = stack.enter_context(apps.tap(selector))
            session.meta.app = selector.label()
            audio.append(("system", bus.source))
            print(f"app:    {selector.label()} (only its audio is recorded)")
        elif not room:
            audio.append(("system", resolve_system(cfg, system)))
            others = [s.app or s.binary for s in apps.playing_streams()]
            if others:
                emit(
                    {
                        "event": "warning",
                        "kind": "other-audio",
                        "message": "Everything the laptop plays is recorded, now also: "
                        + ", ".join(sorted(set(others))),
                    }
                )
        session.meta.tracks = [
            TrackMeta(role=role, file=output.name, stream=i, source=src)
            for i, (role, src) in enumerate(audio)
        ]
        for role, src in audio:
            print(f"{role + ':':8}{src}")

        if not screen:
            print(f"screen: off\nfile:   {output}")
            _capture(session, build_pipeline(audio=audio, output=output), output, (), emit)
            return

        encoder = resolve_encoder(cfg.capture.video_encoder)
        if screen == "monitor":
            print("Privacy: everything shown on the selected monitor is recorded, including notifications.")
        try:
            with _screen_source(screen, cfg.capture.video_fps, window_id, monitor) as (
                source,
                width,
                height,
                source_type,
                fds,
            ):
                video = VideoSpec(source, width, height, cfg.capture.video_fps, encoder)
                session.meta.video = VideoMeta(file=output.name, source_type=source_type, fps=video.fps)
                print(f"screen: {source_type} {video.width}x{video.height} @ {video.fps} fps ({encoder})")
                print(f"file:   {output}")
                _capture(session, build_pipeline(audio=audio, output=output, video=video), output, fds, emit)
        except ScreenCastCancelled:
            session.dir.rmdir()  # nothing was recorded
            raise


def _capture(session: Session, argv: list[str], output: Path, fds: tuple[int, ...], emit: Emit) -> None:
    session.meta.started_at = _now()
    session.save()  # a crash still leaves usable metadata
    roles = [t.role for t in session.meta.tracks]
    health = Health(emit, mic_role=roles[0])
    child = _Child(argv, fds, health, session.work_dir / "capture.log")
    emit({"event": "recording", "dir": str(session.dir), "tracks": roles, "video": bool(session.meta.video)})
    print("Recording. Ctrl+C to stop.", flush=True)
    code = child.run()
    session.meta.ended_at = _now()
    session.meta.mic_dropout_seconds = round(health.dropout_seconds, 1)
    session.save()
    if code != 0:
        raise RuntimeError(
            f"gst-launch-1.0 exited with status {code} (log: {session.work_dir / 'capture.log'})"
        )
    if not output.exists() or output.stat().st_size < MIN_FILE_BYTES:
        raise RuntimeError(f"{output} is missing or empty")
    mapping = map_audio_streams(probe_file(output), roles)
    for track in session.meta.tracks:
        track.stream = mapping[track.role]
    session.save()
    if health.dropout_seconds >= DROPOUT_WARN_SECONDS:
        emit(
            {
                "event": "warning",
                "kind": "mic-dropout-total",
                "message": f"The microphone delivered no audio for {health.dropout_seconds:.0f} s "
                "(muted or dropping; kept as silence).",
            }
        )
