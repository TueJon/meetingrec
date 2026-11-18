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


def test_a_question_or_date_brings_the_next_look_forward():
    h = Hinter(Config(), "call", "Me", workdir=None)
    assert h.due(1000) == ""
    h.add([Line(0, "Remote", "so that is how it went")])
    assert h.due(10) == ""
    assert h.due(30) == "interval"
    h.add([Line(5, "Remote", "Is it released tomorrow?")])
    assert h.due(3) == ""
    assert h.due(10) == "cue"


def _stream(result, *tool_inputs):
    tools = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": used}]}}
        for name, used in tool_inputs
    ]
    return "\n".join(json.dumps(e) for e in [{"type": "system"}, *tools, {"type": "result", **result}])


def test_parse_stream_returns_the_result_and_the_lookups():
    answer = {"type": "tool_use", "name": "StructuredOutput", "input": {"hints": []}}
    stdout = _stream(
        {"is_error": False, "structured_output": {"hints": []}, "total_cost_usd": 0.01},
        ("Grep", {"pattern": "KWEB-1", "path": "notes", "output_mode": "content"}),
        (answer["name"], answer["input"]),
    )
    result, tools = live.parse_stream(stdout)
    assert result["structured_output"] == {"hints": []}
    assert tools == [{"tool": "Grep", "pattern": "KWEB-1", "path": "notes"}]


def test_parse_stream_raises_the_error_of_a_failed_run():
    bad = {"is_error": True, "subtype": "error_during_execution", "errors": ["boom"]}
    for stdout, message in [(_stream(bad), "boom"), ("", "no result")]:
        try:
            live.parse_stream(stdout)
        except RuntimeError as exc:
            assert message in str(exc)
        else:
            raise AssertionError


def test_claude_runs_with_only_the_configured_tools_and_no_user_settings(tmp_path):
    cfg = Config()
    cfg.live.code_dirs = [str(tmp_path)]
    cmd = live.claude_command(cfg, "haiku", "low", live.HINT_SCHEMA)
    assert cmd[cmd.index("--tools") + 1] == "Read,Grep,Glob"
    assert cmd[cmd.index("--permission-mode") + 1] == "default"
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--add-dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--model") + 1] == "haiku"


DRAFTS = [
    {"kind": "answer", "text": "ships 19 Nov", "source": "a.md:3", "source_date": "2026-11-17", "quote": "q"},
    {"kind": "mention", "text": "old plan", "source": "b.md:9", "source_date": "unknown", "quote": "q"},
]


class FakeClaude:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def __call__(self, cmd, prompt, cwd, timeout):
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return live.Call(output, "", [{"tool": "Read", "file_path": "a.md"}], 0.01, 1.5)


def test_look_shows_only_the_drafts_the_check_keeps():
    verdicts = [
        {"index": 0, "keep": True, "reason": "matches", "text": "ships Thu 19 Nov"},
        {"index": 1, "keep": False, "reason": "superseded", "text": ""},
    ]
    fix = {"text": "a.md and b.md disagree on the date", "source": "b.md:9"}
    claude = FakeClaude({"hints": DRAFTS, "record_fixes": [fix]}, {"verdicts": verdicts})
    h = Hinter(Config(), "call", "Me", workdir=None, runner=claude)
    h.add([Line(61, "Remote", "when does it ship?")])
    hints, fixes, entry = h.look()
    assert [hint.render() for hint in hints] == ["[01:01] ANSWER: ships Thu 19 Nov (a.md:3, 2026-11-17)"]
    assert fixes == ["a.md and b.md disagree on the date (b.md:9)"]
    assert h.rejected == ["old plan"]
    assert "when does it ship?" in claude.prompts[1] and "0. ANSWER: ships 19 Nov" in claude.prompts[1]
    assert entry["drafts"] == DRAFTS and entry["verify"]["verdicts"] == verdicts
    assert entry["research"]["tools"] == [{"tool": "Read", "file_path": "a.md"}]
    assert "- old plan" in h.prompt()


def test_look_shows_nothing_when_the_check_fails():
    claude = FakeClaude({"hints": DRAFTS, "record_fixes": []}, RuntimeError("overloaded"))
    h = Hinter(Config(), "call", "Me", workdir=None, runner=claude)
    h.add([Line(0, "Remote", "hello")])
    hints, fixes, entry = h.look()
    assert hints == [] and entry["error"] == "overloaded" and entry["drafts"] == DRAFTS


def test_look_without_a_verify_model_shows_the_drafts():
    cfg = Config()
    cfg.live.verify_model = ""
    claude = FakeClaude({"hints": DRAFTS[1:], "record_fixes": []})
    h = Hinter(cfg, "call", "Me", workdir=None, runner=claude)
    h.add([Line(0, "Remote", "hello")])
    assert [hint.render() for hint in h.look()[0]] == ["[00:00] MENTION: old plan (b.md:9)"]


def test_transcriber_passes_the_glossary_as_hotwords(tmp_path):
    _pcm(tmp_path / "mic.pcm", 5)

    class Model(FakeModel):
        def transcribe(self, audio, **options):
            self.hotwords = options["hotwords"]
            return iter([]), None

    model = Model([])
    Transcriber(model, tmp_path, ["mic"], "en", {"mic": "Me"}, hotwords="Vamsi, PlanRadar").tick()
    assert model.hotwords == "Vamsi, PlanRadar"


def test_review_writes_the_report_next_to_the_hints(tmp_path):
    (tmp_path / "live").mkdir()
    cfg = Config()
    assert live.review(tmp_path, cfg, "call") is None
    (tmp_path / "live" / "hint-log.jsonl").write_text("{}\n")
    seen = []

    def claude(cmd, prompt, cwd, timeout):
        seen.append((cmd, prompt))
        return live.Call({}, "## Verdict\nfine", [], 0.1, 20.0)

    path = live.review(tmp_path, cfg, "call", runner=claude)
    assert path.read_text() == "## Verdict\nfine\n"
    cmd, prompt = seen[0]
    assert cmd[cmd.index("--add-dir") + 1] == str(tmp_path) and "--json-schema" not in cmd
    assert "live/transcript.txt" in prompt
