"""Screen keyframes for the summarizer: frames/kf_<seconds>.jpg from recording.mkv.

One ffmpeg pass decodes the video and yields, per frame, its container timestamp
(= meeting timeline, so a late-starting video stream needs no extra offset), the
scene-change score and a tiny grayscale thumbnail. Selection (scene changes,
a fallback frame every FALLBACK_INTERVAL_S, near-duplicate removal, the cap)
works on that data in Python; only the chosen frames are decoded again at full
resolution.
"""

from __future__ import annotations

import bisect
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .model import Keyframe
from .session import Session

SCENE_THRESHOLD = 0.04  # low: screen content changes little between slides/windows
FALLBACK_INTERVAL_S = 60.0
SETTLE_S = 2.0  # scene hits closer than this form one transition; take the frame after it
DEDUP_MIN_DIFF = 1.0  # mean absolute gray difference (0-255) below which frames are "the same"
MAX_WIDTH = 1600
THUMB_W, THUMB_H = 48, 27

_META = re.compile(r"frame:(\d+)\s+pts:\S+\s+pts_time:(\S+)\s+lavfi\.scene_score=(\S+)")
_NAME = re.compile(r"kf_(\d+)\.jpg$")


@dataclass
class Frame:
    n: int
    t: float
    score: float
    thumb: bytes


def mean_abs_diff(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


# ---- ffmpeg -----------------------------------------------------------------


def scan_video(video: Path, work: Path) -> list[Frame]:
    scores, raw = work / "scene_scores.txt", work / "thumbs.gray"
    chain = (
        f"select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file={scores},"
        f"scale={THUMB_W}:{THUMB_H}:flags=area,format=gray"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-copyts", "-i", str(video), "-an", "-vf", chain,
         "-fps_mode", "passthrough", "-f", "rawvideo", str(raw)],
        check=True,
    )
    meta = _META.findall(scores.read_text())
    data, size = raw.read_bytes(), THUMB_W * THUMB_H
    if len(data) != size * len(meta):
        raise RuntimeError(f"{video.name}: {len(meta)} scored frames but {len(data) // size} thumbnails")
    return [
        Frame(int(n), float(t), float(s), data[i * size : (i + 1) * size])
        for i, (n, t, s) in enumerate(meta)
    ]


def extract_jpegs(video: Path, numbers: list[int], dest: Path) -> list[Path]:
    """Decode the frames with these decode-order numbers at full size; returns files in input order."""
    select = "+".join(f"eq(n,{n})" for n in numbers)
    script = dest / "select.ffscript"
    script.write_text(f"select='{select}',scale='min({MAX_WIDTH},iw)':-2")
    pattern = dest / "tmp_%05d.jpg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-filter_script:v", str(script),
         "-fps_mode", "passthrough", "-q:v", "3", str(pattern)],
        check=True,
    )
    script.unlink()
    files = sorted(dest.glob("tmp_*.jpg"))
    if len(files) != len(numbers):
        raise RuntimeError(f"expected {len(numbers)} frames from {video.name}, got {len(files)}")
    return files


# ---- selection (pure) -------------------------------------------------------


def scene_candidates(frames: list[Frame], threshold: float = SCENE_THRESHOLD) -> list[int]:
    """Indices of the first frame plus the settled frame after each scene transition."""
    picks = {0}
    last_hit: int | None = None
    for i, f in enumerate(frames):
        if f.score < threshold:
            continue
        if last_hit is not None and f.t - frames[last_hit].t > SETTLE_S:
            picks.add(min(last_hit + 1, len(frames) - 1))
        last_hit = i
    if last_hit is not None:
        picks.add(min(last_hit + 1, len(frames) - 1))
    return sorted(picks)


def with_fallback(frames: list[Frame], picks: list[int], interval: float = FALLBACK_INTERVAL_S) -> list[int]:
    """Add a frame wherever two neighbouring picks are more than `interval` seconds apart."""
    times = [f.t for f in frames]
    out = set(picks)
    ends = [*picks[1:], len(frames) - 1]
    for a, b in zip(picks, ends, strict=True):
        t = times[a] + interval
        while t < times[b]:
            out.add(min(bisect.bisect_left(times, t), len(frames) - 1))
            t += interval
    return sorted(out)


def dedupe(frames: list[Frame], picks: list[int], min_diff: float = DEDUP_MIN_DIFF) -> list[int]:
    """Drop picks that look like the last kept one or share its whole second (file name)."""
    kept: list[int] = []
    for i in picks:
        if kept:
            prev = frames[kept[-1]]
            if int(frames[i].t) == int(prev.t) or mean_abs_diff(frames[i].thumb, prev.thumb) < min_diff:
                continue
        kept.append(i)
    return kept


def cap(frames: list[Frame], picks: list[int], limit: int) -> list[int]:
    """Repeatedly drop the pick most redundant with its neighbours, favouring well-spread frames."""
    picks = list(picks)

    def redundancy(k: int) -> float:
        near = [j for j in (k - 1, k + 1) if 0 <= j < len(picks)]
        diff = min(mean_abs_diff(frames[picks[k]].thumb, frames[picks[j]].thumb) for j in near)
        gap = min(abs(frames[picks[k]].t - frames[picks[j]].t) for j in near)
        return diff * (1 + gap / FALLBACK_INTERVAL_S)

    while len(picks) > max(limit, 1):
        del picks[min(range(len(picks)), key=redundancy)]
    return picks


# ---- entry point ------------------------------------------------------------


def _existing(frames_dir: Path, video: Path, limit: int) -> list[Keyframe] | None:
    files = sorted(frames_dir.glob("kf_*.jpg"), key=lambda p: int(_NAME.search(p.name)[1]))
    video_mtime = video.stat().st_mtime
    if not files or len(files) > limit or any(f.stat().st_mtime < video_mtime for f in files):
        return None
    return [Keyframe(float(_NAME.search(f.name)[1]), f"{frames_dir.name}/{f.name}") for f in files]


def extract_keyframes(session: Session, cfg: Config) -> list[Keyframe]:
    if session.meta.video is None:
        return []
    video = session.dir / session.meta.video.file
    frames_dir = session.dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    limit = cfg.summary.max_keyframes
    if (reused := _existing(frames_dir, video, limit)) is not None:
        return reused

    frames = scan_video(video, session.work_dir)
    if not frames:
        return []
    picks = dedupe(frames, with_fallback(frames, scene_candidates(frames)))
    picks = cap(frames, picks, limit)

    for old in frames_dir.glob("kf_*.jpg"):
        old.unlink()
    files = extract_jpegs(video, [frames[i].n for i in picks], frames_dir)
    keyframes = []
    for i, tmp in zip(picks, files, strict=True):
        seconds = int(frames[i].t)
        tmp.rename(frames_dir / f"kf_{seconds}.jpg")
        keyframes.append(Keyframe(float(seconds), f"frames/kf_{seconds}.jpg"))
    return keyframes
