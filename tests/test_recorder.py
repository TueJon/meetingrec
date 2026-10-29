from pathlib import Path

import pytest

from meetingrec.capture import recorder, x11
from meetingrec.capture.recorder import VideoSpec, build_pipeline

OUT = Path("/tmp/x/recording.mka")


AUDIO = [("mic", "m"), ("system", "s")]


def test_audio_only_pipeline_has_two_tagged_flac_branches():
    argv = build_pipeline(audio=[("mic", "mic0"), ("system", "sink0.monitor")], output=OUT)
    assert argv[:3] == ["gst-launch-1.0", "-m", "-e"]
    assert f"location={OUT}" in argv
    assert argv.count("flacenc") == 2
    assert argv.count("audio/x-raw,rate=48000,channels=1") == 2
    assert argv.index("tags=title=mic") < argv.index("tags=title=system")
    assert argv.index("device=mic0") < argv.index("device=sink0.monitor")
    assert argv.count("mux.") == 2
    assert "pipewiresrc" not in argv


def test_audio_branch_keeps_the_timeline_and_reports_levels():
    argv = build_pipeline(audio=[("room", "mic0")], output=OUT)
    assert argv.count("pulsesrc") == 1
    assert "tags=title=room" in argv
    assert f"buffer-time={recorder.PULSE_BUFFER_US}" in argv
    assert (
        argv.index("audio/x-raw,rate=48000,channels=1")
        < argv.index("audiorate")
        < argv.index("name=level_room")
        < argv.index("flacenc")
    )


def test_video_branch_uses_even_fixed_caps_and_keepalive():
    source = recorder.pipewire_source(fd=7, node_id=42, fps=2)
    video = VideoSpec(source=source, width=1281, height=721, fps=2, encoder="x264enc")
    argv = build_pipeline(audio=AUDIO, output=OUT, video=video)
    assert argv.count("mux.") == 3
    assert {"pipewiresrc", "fd=7", "path=42", "keepalive-time=500"} <= set(argv)
    assert "video/x-raw,framerate=2/1" in argv
    assert "video/x-raw,width=1280,height=720,pixel-aspect-ratio=1/1" in argv
    assert "add-borders=true" in argv
    assert (
        argv.index("videorate") < argv.index("videoscale") < argv.index("x264enc") < argv.index("h264parse")
    )


def test_ximagesrc_source_is_used_as_given():
    video = VideoSpec(x11.window_source(77), 640, 480, 2, "nvh264enc")
    argv = build_pipeline(audio=AUDIO, output=OUT, video=video)
    assert argv[argv.index("ximagesrc") : argv.index("ximagesrc") + 4] == [
        "ximagesrc",
        "xid=77",
        "use-damage=false",
        "show-pointer=true",
    ]
    assert "pipewiresrc" not in argv
    assert argv.index("ximagesrc") < argv.index("videoconvert") < argv.index("nvh264enc")


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
    assert recorder.map_audio_streams(probe, ["mic", "system"]) == {"mic": 1, "system": 0}


def test_map_audio_streams_missing_track():
    with pytest.raises(RuntimeError, match="system"):
        recorder.map_audio_streams({"streams": [_stream("audio", "mic")]}, ["mic", "system"])


def test_map_audio_streams_room_only():
    assert recorder.map_audio_streams({"streams": [_stream("audio", "room")]}, ["room"]) == {"room": 0}


LEVEL = (
    'Got message #48 from element "level_mic" (element): level, endtime=(guint64)500528687, '
    "timestamp=(guint64)528687, duration=(guint64)500000000, rms=(GValueArray)< -44.5 >, "
    "peak=(GValueArray)< -20.25 >, decay=(GValueArray)< -20.25 >;"
)


def test_parse_level():
    assert recorder.parse_level(LEVEL) == ("mic", -44.5, -20.25)
    assert recorder.parse_level(LEVEL.replace("< -44.5 >", "< -inf >"))[1] == float("-inf")
    assert recorder.parse_level('Got message #12 from element "level_mic" (state-changed): x') is None


def test_health_warns_once_per_burst_of_mic_dropouts():
    events = []
    health = recorder.Health(events.append, mic_role="mic")
    for i in range(10):
        health.level("system", -700.0, -350.0, now=i * 0.5)  # a silent call is not a dropout
    assert health.dropout_seconds == 0
    for i in range(10):
        health.level("mic", -700.0, -350.0, now=i * 0.5)
    warnings = [e for e in events if e["event"] == "warning"]
    assert len(warnings) == 1 and warnings[0]["kind"] == "mic-dropout"
    assert health.dropout_seconds == 5.0
    health.level("mic", -30.0, -10.0, now=6.0)
    assert health.dropout_seconds == 5.0
