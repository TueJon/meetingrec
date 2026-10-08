import subprocess
from pathlib import Path

import pytest

from meetingrec import visual
from meetingrec.config import Config
from meetingrec.session import MeetingMeta, Session, VideoMeta

pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0, reason="ffmpeg missing"
)

OFFSET = 5.0


def make_video(path: Path, offset: float) -> None:
    """3 min at 2 fps: red 0-40 s, blue 40-60, green 60-150 (white box appears at 100 s), yellow 150-180."""
    sources = [
        "color=c=red:s=640x360:r=2:d=40",
        "color=c=blue:s=640x360:r=2:d=20",
        "color=c=green:s=640x360:r=2:d=90,drawbox=x=300:y=150:w=60:h=40:c=white:t=fill:enable='gte(t,40)'",
        "color=c=yellow:s=640x360:r=2:d=30",
    ]
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for s in sources:
        cmd += ["-f", "lavfi", "-i", s]
    cmd += ["-filter_complex", "concat=n=4:v=1:a=0", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-output_ts_offset", str(offset), str(path)]
    subprocess.run(cmd, check=True)


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("vid") / "recording.mkv"
    make_video(path, OFFSET)
    return path


def make_session(tmp_path: Path, video: Path) -> Session:
    (tmp_path / "recording.mkv").symlink_to(video)
    meta = MeetingMeta(title="t", started_at="2026-09-14T10:00:00+02:00",
                       video=VideoMeta("recording.mkv", "monitor", 2))
    return Session(tmp_path, meta)


def cfg_with(limit: int) -> Config:
    cfg = Config()
    cfg.summary.max_keyframes = limit
    return cfg


def test_scan_finds_scene_scores_with_offset(video, tmp_path):
    frames = visual.scan_video(video, tmp_path)
    assert len(frames) == 360
    assert frames[0].t == pytest.approx(OFFSET)  # container timestamps include the stream start
    hits = [f.t for f in frames if f.score >= visual.SCENE_THRESHOLD]
    assert hits == pytest.approx([45.0, 65.0, 155.0])


def test_selection_pipeline(video, tmp_path):
    frames = visual.scan_video(video, tmp_path)
    scene = visual.scene_candidates(frames)
    assert [frames[i].t for i in scene] == pytest.approx([5.0, 45.5, 65.5, 155.5])
    with_fb = visual.with_fallback(frames, scene)
    # 65.5 -> 155.5 is 90 s of static screen: one fallback frame in between
    assert len(with_fb) == 5
    fallback = [i for i in with_fb if i not in scene]
    assert frames[fallback[0]].t == pytest.approx(125.5, abs=1.0)
    assert visual.dedupe(frames, with_fb) == with_fb  # the box makes the fallback frame distinct


def test_dedupe_drops_identical_fallback_frames(video, tmp_path):
    frames = visual.scan_video(video, tmp_path)
    static = [i for i, f in enumerate(frames) if 70 <= f.t <= 90]  # plain green, no box yet
    kept = visual.dedupe(frames, [static[0], static[10], static[30]])
    assert kept == [static[0]]


def test_cap_keeps_most_distinct_in_order(video, tmp_path):
    frames = visual.scan_video(video, tmp_path)
    picks = visual.with_fallback(frames, visual.scene_candidates(frames))
    capped = visual.cap(frames, picks, 3)
    assert len(capped) == 3 and capped == sorted(capped)
    colours = {frames[i].thumb[:1] for i in capped}
    assert len(colours) >= 3  # three different screens survive, not three greens


def test_fallback_interval_pure():
    frames = [visual.Frame(i, i * 0.5, 0.0, bytes(4)) for i in range(2400)]  # 20 min static
    picks = visual.with_fallback(frames, [0])
    assert [frames[i].t for i in picks[:4]] == [0.0, 60.0, 120.0, 180.0]
    assert len(picks) == 20


def test_extract_keyframes_end_to_end(video, tmp_path):
    session = make_session(tmp_path, video)
    kfs = visual.extract_keyframes(session, cfg_with(12))
    assert [k.time for k in kfs] == [5, 45, 65, 125, 155]
    assert all(k.path == f"frames/kf_{int(k.time)}.jpg" for k in kfs)
    first = tmp_path / kfs[0].path
    assert first.stat().st_size > 0
    assert {p.name for p in (tmp_path / "frames").iterdir()} == {f"kf_{int(k.time)}.jpg" for k in kfs}
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width", "-of", "csv=p=0", str(first)], text=True
    )
    assert int(probe) == 640  # never upscaled


def test_extract_keyframes_cap_and_reuse(video, tmp_path):
    session = make_session(tmp_path, video)
    kfs = visual.extract_keyframes(session, cfg_with(3))
    assert len(kfs) == 3 and [k.time for k in kfs] == sorted(k.time for k in kfs)
    mtimes = [(tmp_path / k.path).stat().st_mtime_ns for k in kfs]
    again = visual.extract_keyframes(session, cfg_with(3))
    assert again == kfs
    assert [(tmp_path / k.path).stat().st_mtime_ns for k in again] == mtimes  # reused, not rewritten
    assert len(visual.extract_keyframes(session, cfg_with(2))) == 2  # lower cap: re-extract


def test_no_video_means_no_keyframes(tmp_path):
    meta = MeetingMeta(title="t", started_at="2026-09-14T10:00:00+02:00")
    assert visual.extract_keyframes(Session(tmp_path, meta), Config()) == []
