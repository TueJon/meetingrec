import json
from dataclasses import dataclass

import numpy as np

from meetingrec import live
from meetingrec.config import Config
from meetingrec.live import BYTES_PER_SECOND, Hinter, Line, Transcriber


@dataclass
class Seg:
    start: float
    end: float
    text: str


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.seconds = []

    def transcribe(self, audio, **_):
        self.seconds.append(len(audio) / live.RATE)
        return iter(self.segments), None


def _pcm(path, seconds):
    path.write_bytes(np.zeros(int(seconds * live.RATE), np.int16).tobytes())


def test_transcriber_keeps_a_cut_last_segment_for_the_next_chunk(tmp_path):
    _pcm(tmp_path / "system.pcm", 10)
    model = FakeModel([Seg(1.0, 4.0, " Hallo zusammen."), Seg(8.0, 9.7, " und dann")])
    t = Transcriber(model, tmp_path, ["system"], "de", {"system": "Remote"}, {"system": 0})
    assert t.tick() == [Line(1.0, "Remote", "Hallo zusammen.")]
    assert t.offsets["system"] == 8 * BYTES_PER_SECOND
    _pcm(tmp_path / "system.pcm", 20)
    model.segments = [Seg(0.0, 3.0, " und dann gehen wir.")]
    assert t.tick() == [Line(8.0, "Remote", "und dann gehen wir.")]
    assert model.seconds == [10.0, 12.0]


def test_transcriber_waits_for_enough_audio(tmp_path):
    _pcm(tmp_path / "mic.pcm", 1)
    model = FakeModel([])
    t = Transcriber(model, tmp_path, ["mic", "system"], "auto", {"mic": "Me", "system": "Remote"})
    assert t.tick() == []
    assert model.seconds == []


def test_prompt_marks_new_lines_and_drops_old_ones():
    cfg = Config()
    cfg.live.window_minutes = 1
    h = Hinter(cfg, "call", "Me", workdir=None)
    h.add([Line(0, "Me", "too old"), Line(50, "Remote", "seen"), Line(100, "Remote", "fresh")])
    h.seen = 2
    prompt = h.prompt()
    assert "too old" not in prompt
    assert prompt.index("Remote: seen") < prompt.index("\n>>> new\n") < prompt.index("Remote: fresh")
    assert h.due()


def test_parse_hints():
    ok = {"is_error": False, "structured_output": {"hints": [{"kind": "ask", "text": "t", "source": "s"}]}}
    assert live.parse_hints(json.dumps(ok)) == [{"kind": "ask", "text": "t", "source": "s"}]
    bad = {"is_error": True, "subtype": "error_during_execution", "errors": ["boom"]}
    try:
        live.parse_hints(json.dumps(bad))
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError


def test_claude_runs_with_only_the_configured_tools():
    cmd = live.claude_command(Config())
    assert cmd[cmd.index("--tools") + 1] == "Read,Grep,Glob"
    assert cmd[cmd.index("--permission-mode") + 1] == "default"
    assert cmd[cmd.index("--model") + 1] == "haiku"
