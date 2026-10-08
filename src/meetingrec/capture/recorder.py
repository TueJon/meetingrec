"""Record mic + system audio (+ optional screen) into one Matroska file with GStreamer.

One `gst-launch-1.0` process owns every track, so all of them share one clock and
start at the same instant: that is what keeps audio and video in sync.
"""

from __future__ import annotations

import datetime as dt
import json
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from ..config import Config
from ..session import Session, TrackMeta, VideoMeta
from .portal import ScreenCastCancelled, restore_token_path, screencast

AUDIO_ROLES = ("mic", "system")
FINALIZE_TIMEOUT = 30
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


def resolve_sources(cfg: Config, mic: str | None, system: str | None) -> tuple[str, str]:
    available = list_sources()
    default_mic = _pactl("get-default-source").strip()
    default_monitor = _pactl("get-default-sink").strip() + ".monitor"
    return (
        resolve_source(mic, cfg.capture.mic, default_mic, available, "mic"),
        resolve_source(system, cfg.capture.system, default_monitor, available, "system"),
    )


# ---- pipeline ---------------------------------------------------------------


@dataclass
class VideoSpec:
    fd: int
    node_id: int
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
        "!",
        "queue",
        "!",
        "audioconvert",
        "!",
        "audioresample",
        "!",
        "audio/x-raw,rate=48000,channels=1",
        "!",
        "flacenc",
        "!",
        "taginject",
        f"tags=title={role}",
        "!",
        "mux.",
    ]


def _video_branch(video: VideoSpec, source: list[str]) -> list[str]:
    w, h = _even(video.width), _even(video.height)
    return [
        *source,
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
    mic: str,
    system: str,
    output: Path,
    video: VideoSpec | None = None,
    _video_source: list[str] | None = None,
) -> list[str]:
    """gst-launch-1.0 argv. `_video_source` replaces pipewiresrc (tests only)."""
    argv = ["gst-launch-1.0", "-q", "-e", "matroskamux", "name=mux", "!", "filesink", f"location={output}"]
    argv += _audio_branch(mic, "mic") + _audio_branch(system, "system")
    if video:
        source = _video_source or [
            "pipewiresrc",
            f"fd={video.fd}",
            f"path={video.node_id}",
            "do-timestamp=true",
            f"keepalive-time={1000 // video.fps}",
        ]
        argv += _video_branch(video, source)
    return argv


# ---- stream mapping -----------------------------------------------------------


def map_audio_streams(probe: dict) -> dict[str, int]:
    """Role -> 0-based index among the audio streams, from ffprobe's title tags."""
    audio = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    # Matroska tags come back upper-cased ("TITLE") from ffprobe.
    mapping = {
        {k.lower(): v for k, v in s.get("tags", {}).items()}.get("title"): i for i, s in enumerate(audio)
    }
    missing = [r for r in AUDIO_ROLES if r not in mapping]
    if missing:
        raise RuntimeError(f"recording lacks audio track(s): {', '.join(missing)}")
    return {r: mapping[r] for r in AUDIO_ROLES}


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


# ---- process control ----------------------------------------------------------


class _Child:
    """gst-launch in its own session, so the terminal's Ctrl+C reaches only us and
    the child gets exactly one SIGINT (which `-e` turns into EOS)."""

    def __init__(self, argv: list[str], fds: tuple[int, ...]) -> None:
        self.proc = subprocess.Popen(argv, pass_fds=fds, start_new_session=True)
        self._stopped = False
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: self.stop())

    def stop(self) -> None:
        if not self._stopped and self.proc.poll() is None:
            self._stopped = True
            self.proc.send_signal(signal.SIGINT)

    def run(self) -> int:
        try:
            self.proc.wait()
        except KeyboardInterrupt:
            self.stop()
        return self._finalize()

    def _finalize(self) -> int:
        if self._stopped:
            print("Stopping, finalizing the file ...", flush=True)
        while True:
            try:
                return self.proc.wait(timeout=FINALIZE_TIMEOUT)
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


def record(session: Session, cfg: Config, *, screen: str | None, mic: str | None, system: str | None) -> None:
    mic_src, system_src = resolve_sources(cfg, mic, system)
    output = session.path("recording.mkv" if screen else "recording.mka")
    session.meta.recorder_version = __version__
    session.meta.tracks = [
        TrackMeta(role=role, file=output.name, stream=i, source=src)
        for i, (role, src) in enumerate(zip(AUDIO_ROLES, (mic_src, system_src), strict=True))
    ]
    print(f"mic:    {mic_src}\nsystem: {system_src}")

    if not screen:
        print(f"screen: off\nfile:   {output}")
        _capture(session, build_pipeline(mic=mic_src, system=system_src, output=output), output, ())
        return

    encoder = resolve_encoder(cfg.capture.video_encoder)
    if screen == "monitor":
        print("Privacy: everything shown on the selected monitor is recorded, including notifications.")
    try:
        with screencast(screen, restore_token_path()) as sc:
            video = VideoSpec(
                sc.fd, sc.node_id, sc.width or 1920, sc.height or 1080, cfg.capture.video_fps, encoder
            )
            session.meta.video = VideoMeta(file=output.name, source_type=sc.source_type, fps=video.fps)
            print(f"screen: {sc.source_type} {video.width}x{video.height} @ {video.fps} fps ({encoder})")
            print(f"file:   {output}")
            argv = build_pipeline(mic=mic_src, system=system_src, output=output, video=video)
            _capture(session, argv, output, (sc.fd,))
    except ScreenCastCancelled:
        session.dir.rmdir()  # nothing was recorded
        raise


def _capture(session: Session, argv: list[str], output: Path, fds: tuple[int, ...]) -> None:
    session.meta.started_at = _now()
    session.save()  # a crash still leaves usable metadata
    print("Recording. Ctrl+C to stop.", flush=True)
    code = _Child(argv, fds).run()
    session.meta.ended_at = _now()
    session.save()
    if code != 0:
        raise RuntimeError(f"gst-launch-1.0 exited with status {code}")
    if not output.exists() or output.stat().st_size < MIN_FILE_BYTES:
        raise RuntimeError(f"{output} is missing or empty")
    mapping = map_audio_streams(probe_file(output))
    for track in session.meta.tracks:
        track.stream = mapping[track.role]
    session.save()
