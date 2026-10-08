"""A meeting directory and its metadata (meeting.json).

Layout written by `meetingrec record`:

    <out_root>/<YYYY-MM-DD_HHMMSS>_<slug>/
        meeting.json        metadata (this module)
        recording.mkv       screen video + audio tracks  (or recording.mka, audio only)
        notes.md            optional, user-written; authoritative for the summary
        work/               derived 16 kHz WAVs per track (safe to delete)
        transcript.*        render.py
        summary.{json,md}   summarize.py
        frames/             visual.py keyframes

Directories from meetingrec 0.1 hold only audio.wav (mic + system mixed to mono);
they load as a single "mixed" track.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .model import TrackRole

META_FILE = "meeting.json"
LEGACY_AUDIO = "audio.wav"
_DIR_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})")


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", text) or "meeting"


@dataclass
class TrackMeta:
    role: TrackRole
    file: str  # relative to the meeting directory
    stream: int | None = None  # audio stream index inside `file` (ffmpeg 0:a:<n>)
    source: str = ""  # capture device name, informational


@dataclass
class VideoMeta:
    file: str
    source_type: str  # "window" | "monitor"
    fps: int


@dataclass
class MeetingMeta:
    title: str
    started_at: str  # ISO 8601 with UTC offset
    ended_at: str | None = None
    tracks: list[TrackMeta] = field(default_factory=list)
    video: VideoMeta | None = None
    recorder_version: str = ""

    @property
    def started(self) -> dt.datetime:
        return dt.datetime.fromisoformat(self.started_at)


@dataclass
class Session:
    dir: Path
    meta: MeetingMeta

    # ---- construction ------------------------------------------------------

    @classmethod
    def create(cls, out_root: Path, title: str, started: dt.datetime) -> Session:
        d = out_root / f"{started:%Y-%m-%d_%H%M%S}_{slugify(title)}"
        d.mkdir(parents=True, exist_ok=False)
        return cls(d, MeetingMeta(title=title, started_at=started.isoformat(timespec="seconds")))

    @classmethod
    def load(cls, d: Path) -> Session:
        d = d.resolve()
        if d.is_file():
            d = d.parent
        meta_path = d / META_FILE
        if meta_path.exists():
            raw = json.loads(meta_path.read_text())
            raw["tracks"] = [TrackMeta(**t) for t in raw.get("tracks", [])]
            raw["video"] = VideoMeta(**raw["video"]) if raw.get("video") else None
            return cls(d, MeetingMeta(**raw))
        if (d / LEGACY_AUDIO).exists():
            return cls(d, _legacy_meta(d))
        raise FileNotFoundError(f"{d} is not a meeting directory (no {META_FILE} or {LEGACY_AUDIO})")

    def save(self) -> None:
        (self.dir / META_FILE).write_text(json.dumps(asdict(self.meta), indent=2, ensure_ascii=False) + "\n")

    # ---- paths -------------------------------------------------------------

    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def work_dir(self) -> Path:
        p = self.dir / "work"
        p.mkdir(exist_ok=True)
        return p

    def track(self, role: TrackRole) -> TrackMeta | None:
        return next((t for t in self.meta.tracks if t.role == role), None)

    def track_audio(self, role: TrackRole) -> tuple[Path, float]:
        """16 kHz mono WAV of one track, plus its start offset on the meeting timeline.

        Matroska tracks start at slightly different timestamps (tens of ms); add
        the returned offset to every time measured inside the WAV.
        """
        t = self.track(role)
        if t is None:
            raise KeyError(f"{self.dir.name} has no {role} track")
        src = self.dir / t.file
        if t.stream is None:  # already a plain audio file (legacy audio.wav)
            return src, 0.0
        out = self.work_dir / f"{role}.wav"
        offset = _stream_start(src, t.stream)
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(src),
                    "-map",
                    f"0:a:{t.stream}",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(out),
                ],
                check=True,
            )
        return out, offset


def _stream_start(path: Path, stream: int) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            f"a:{stream}",
            "-show_entries",
            "stream=start_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _legacy_meta(d: Path) -> MeetingMeta:
    m = _DIR_TS.match(d.name)
    if m:
        started = dt.datetime.strptime(f"{m[1]} {m[2]}", "%Y-%m-%d %H%M%S").astimezone()
        title = d.name[m.end() :].lstrip("_") or "meeting"
    else:
        started = dt.datetime.fromtimestamp((d / LEGACY_AUDIO).stat().st_mtime).astimezone()
        title = d.name
    return MeetingMeta(
        title=title,
        started_at=started.isoformat(timespec="seconds"),
        tracks=[TrackMeta(role="mixed", file=LEGACY_AUDIO)],
    )
