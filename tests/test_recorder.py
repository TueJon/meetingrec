from pathlib import Path

import pytest

from meetingrec.capture import recorder
from meetingrec.capture.recorder import VideoSpec, build_pipeline

OUT = Path("/tmp/x/recording.mka")


def test_audio_only_pipeline_has_two_tagged_flac_branches():
    argv = build_pipeline(mic="mic0", system="sink0.monitor", output=OUT)
    assert argv[:3] == ["gst-launch-1.0", "-q", "-e"]
    assert f"location={OUT}" in argv
    assert argv.count("flacenc") == 2
    assert argv.count("audio/x-raw,rate=48000,channels=1") == 2
    assert argv.index("tags=title=mic") < argv.index("tags=title=system")
    assert argv.index("device=mic0") < argv.index("device=sink0.monitor")
    assert argv.count("mux.") == 2
    assert "pipewiresrc" not in argv


def test_video_branch_uses_even_fixed_caps_and_keepalive():
    video = VideoSpec(fd=7, node_id=42, width=1281, height=721, fps=2, encoder="x264enc")
    argv = build_pipeline(mic="m", system="s", output=OUT, video=video)
    assert argv.count("mux.") == 3
    assert {"pipewiresrc", "fd=7", "path=42", "keepalive-time=500"} <= set(argv)
    assert "video/x-raw,framerate=2/1" in argv
    assert "video/x-raw,width=1280,height=720,pixel-aspect-ratio=1/1" in argv
    assert "add-borders=true" in argv
    assert (
        argv.index("videorate") < argv.index("videoscale") < argv.index("x264enc") < argv.index("h264parse")
    )


def test_video_source_override_replaces_pipewiresrc():
    video = VideoSpec(0, 0, 640, 480, 2, "nvh264enc")
    argv = build_pipeline(
        mic="m", system="s", output=OUT, video=video, _video_source=["videotestsrc", "is-live=true"]
    )
    assert "videotestsrc" in argv
    assert "pipewiresrc" not in argv


def test_encoder_args_keyframe_every_ten_seconds():
    assert "gop-size=20" in recorder.encoder_args("nvh264enc", 2)
    x264 = recorder.encoder_args("x264enc", 2)
    assert {"key-int-max=20", "tune=stillimage", "speed-preset=veryfast", "threads=1"} <= set(x264)
    with pytest.raises(ValueError):
        recorder.encoder_args("vp9enc", 2)


def test_explicit_encoder_skips_probe(monkeypatch):
    monkeypatch.setattr(recorder, "nvenc_works", lambda: pytest.fail("probe must not run"))
    assert recorder.resolve_encoder("x264enc") == "x264enc"


@pytest.mark.parametrize("works,expected", [(True, "nvh264enc"), (False, "x264enc")])
def test_auto_encoder(monkeypatch, works, expected):
    monkeypatch.setattr(recorder, "nvenc_works", lambda: works)
    assert recorder.resolve_encoder("auto") == expected


def test_resolve_source_precedence_and_error():
    avail = ["a", "b", "c"]
    assert recorder.resolve_source("c", "b", "a", avail, "mic") == "c"
    assert recorder.resolve_source(None, "b", "a", avail, "mic") == "b"
    assert recorder.resolve_source(None, "", "a", avail, "mic") == "a"
    with pytest.raises(RuntimeError, match="Available sources:\n  a\n  b"):
        recorder.resolve_source("zzz", "", "a", avail, "mic")


def _stream(kind, title=None):
    return {"codec_type": kind, **({"tags": {"TITLE": title}} if title else {})}


def test_map_audio_streams_counts_only_audio_streams():
    probe = {"streams": [_stream("video"), _stream("audio", "system"), _stream("audio", "mic")]}
    assert recorder.map_audio_streams(probe) == {"mic": 1, "system": 0}


def test_map_audio_streams_missing_track():
    with pytest.raises(RuntimeError, match="system"):
        recorder.map_audio_streams({"streams": [_stream("audio", "mic")]})
