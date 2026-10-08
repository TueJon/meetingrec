import datetime as dt
import json
import subprocess

import pytest

from meetingrec import summarize as sm
from meetingrec.config import Config
from meetingrec.model import Keyframe, Segment, SpeakerInfo, Transcript
from meetingrec.session import MeetingMeta, Session

MONDAY = dt.date(2026, 9, 14)


def make_transcript(language: str = "de") -> Transcript:
    segs = [
        Segment(0, 10, "Guten Morgen, wir starten mit dem Release.", "system", "SPEAKER_00"),
        Segment(12, 30, "Ich kümmere mich um die Release Notes bis nächste Woche Donnerstag.", "mic", "self"),
        Segment(40, 60, "Wer prüft den Server? Das ist noch offen.", "system", "SPEAKER_01"),
        Segment(70, 90, "Anna macht das Deployment am Freitag.", "system", "SPEAKER_00"),
    ]
    speakers = {
        "SPEAKER_00": SpeakerInfo("SPEAKER_00", "Anna Berger", "voiceprint"),
        "SPEAKER_01": SpeakerInfo("SPEAKER_01"),
        "self": SpeakerInfo("self", "Jonas", "self"),
    }
    return Transcript(language, segs, speakers, asr_model="large-v3")


def make_session(tmp_path, started="2026-09-14T10:00:00+02:00") -> Session:
    return Session(tmp_path, MeetingMeta(title="Release Sync", started_at=started))


def cfg_with(**kw) -> Config:
    cfg = Config(self_name="Jonas", participants=["Bernd Huber"])
    for k, v in kw.items():
        setattr(cfg.summary, k, v)
    return cfg


def test_calendar_lines():
    lines = sm.calendar_lines(MONDAY, "de")
    assert lines[0] == "Mo 2026-09-07"
    assert lines[7] == "Mo 2026-09-14  <- meeting day"
    assert "Do 2026-09-17" in lines
    assert lines[-1].startswith("Mo 2026-10-26")
    assert len(lines) == 7 + 42 + 1
    assert sm.calendar_lines(MONDAY, "en")[8] == "Tue 2026-09-15"


def test_oct24_saturday_case_is_flagged():
    assert dt.date(2026, 10, 24).weekday() == 5
    assert sm.date_problem("2026-10-24", "nächste Woche Donnerstag", MONDAY)
    assert not sm.date_problem("2026-09-24", "nächste Woche Donnerstag", MONDAY)


@pytest.mark.parametrize(
    "date,phrase,bad",
    [
        ("2026-09-17", "Do", False),
        ("2026-09-18", "Do.", True),
        ("2026-09-17", "next Thu", False),
        ("2026-09-17", "Thursday or Friday", False),
        ("2026-09-19", "Thursday or Friday", True),
        ("2026-09-13", None, True),  # before the meeting
        ("14.09.2026", None, True),  # unparsable
        ("2026-09-20", "So bald wie möglich", False),  # bare "So" is a word, not Sunday
        ("2026-09-20", "bis Sonntag", False),
        ("2026-09-20", "ohne Wochentag", False),
        (None, "Montag", False),
    ],
)
def test_date_problem(date, phrase, bad):
    assert sm.date_problem(date, phrase, MONDAY) is bad


def test_named_weekdays_whole_words_only():
    assert sm.named_weekdays("Donnerstagabend") == set()
    assert sm.named_weekdays("am Fr oder Mo.") == {4, 0}


@pytest.mark.parametrize(
    "owner,expected",
    [
        ("anna berger", "Anna Berger"),
        ("Anna", "Anna Berger"),
        ("Anna Meier", None),
        ("Bernd", "Bernd Huber"),
        ("Peter", None),
        (None, None),
        ("", None),
    ],
)
def test_match_owner(owner, expected):
    assert sm.match_owner(owner, ["Anna Berger", "Bernd Huber", "Jonas"]) == expected


def test_match_owner_ambiguous_first_name():
    assert sm.match_owner("Anna", ["Anna Berger", "Anna Huber"]) is None


def test_quote_found_fuzzy_and_missing():
    text = "Ich kümmere mich um die Release Notes bis nächste Woche Donnerstag."
    assert sm.quote_found("Release Notes bis nächste Woche Donnerstag", text)
    assert sm.quote_found("kümmere mich um die Release-Notes bis nächste Woche Donerstag", text)
    assert not sm.quote_found("wir verschieben alles auf Oktober", text)
    assert not sm.quote_found("", text)


def test_quote_window_limits_search():
    t = make_transcript()
    assert "Release Notes" in sm.text_near(t, 20)
    assert "Deployment" not in sm.text_near(t, 20, radius=10)
    assert "Deployment" in sm.text_near(t, None)


def test_parse_ref():
    assert sm.parse_ref("01:30") == 90
    assert sm.parse_ref("[1:02:03]") == 3723
    assert sm.parse_ref("soon") is None


def model_output(**over) -> dict:
    data = {
        "summary": "Das Release wurde besprochen.",
        "key_points": [{"text": "Release startet", "refs": ["00:00"]}],
        "decisions": [],
        "action_items": [
            {
                "task": "Release Notes schreiben", "owner": "Jonas",
                "due_phrase": "nächste Woche Donnerstag", "due_date": "2026-09-24",
                "quote": "Release Notes bis nächste Woche Donnerstag", "ref": "00:12",
            },
            {
                "task": "Server prüfen", "owner": "Peter", "due_phrase": "nächste Woche Donnerstag",
                "due_date": "2026-10-24", "quote": "Server prüfen lassen", "ref": "09:99:99",
            },
        ],
        "open_questions": [{"text": "Wer prüft den Server?", "refs": ["00:40", "30:00"]}],
    }
    return {**data, **over}


def test_validate_flags_everything():
    t = make_transcript()
    allowed, _ = sm.speaker_allow_list(t, cfg_with())
    res = sm.validate(model_output(), t, allowed, MONDAY, "de")
    good, bad = res["action_items"]
    assert good["warnings"] == [] and good["owner"] == "Jonas" and good["due_date"] == "2026-09-24"
    assert bad["owner"] is None and bad["due_date"] is None
    assert bad["due_phrase"] == "nächste Woche Donnerstag"
    assert len(bad["warnings"]) == 4  # ref, owner, date, quote
    assert len(res["open_questions"][0]["warnings"]) == 1  # 30:00 beyond a 90 s transcript
    assert any("Peter" in w for w in sm.collect_warnings(res, "de"))


def test_allow_list_and_unknown_speakers():
    allowed, unknown = sm.speaker_allow_list(make_transcript(), cfg_with())
    assert allowed == ["Anna Berger", "Jonas", "Bernd Huber"]
    assert unknown == ["SPEAKER_01"]


def test_prompt_context(tmp_path):
    session = make_session(tmp_path)
    (tmp_path / "notes.md").write_text("Anna ist die Projektleiterin.")
    cfg = cfg_with()
    cfg.glossary = ["Kubernetes"]
    prompt = sm.build_prompt(session, make_transcript(), cfg, [Keyframe(65, "frames/kf_65.jpg")], "de")
    for needle in (
        "Monday 2026-09-14 10:00 (UTC+02:00)", "Do 2026-09-17", "Anna Berger, Jonas, Bernd Huber",
        "SPEAKER_01", "Kubernetes", "Authoritative", "Anna ist die Projektleiterin.",
        "[01:05] frames/kf_65.jpg", "[00:12] Jonas: Ich kümmere", "Write all text in German",
    ):
        assert needle in prompt, needle
    assert "Screen frames" not in sm.build_prompt(session, make_transcript(), cfg, [], "de")


def test_timezone_conversion(tmp_path):
    cfg = cfg_with()
    cfg.timezone = "America/New_York"
    start = sm.meeting_start(make_session(tmp_path), cfg)
    assert (start.hour, sm.utc_offset(start)) == (4, "UTC-04:00")


def test_language_resolution():
    assert sm.resolve_language(Config(), make_transcript("de")) == "de"
    assert sm.resolve_language(Config(), make_transcript("fr")) == "en"
    cfg = cfg_with(language="en")
    assert sm.resolve_language(cfg, make_transcript("de")) == "en"


def proc(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_parse_envelope_ok():
    env = {"type": "result", "subtype": "success", "is_error": False,
           "structured_output": {"summary": "x"}, "modelUsage": {"claude-opus-5-5": {}}}
    got = sm.parse_envelope(proc(json.dumps(env)))
    assert got.data == {"summary": "x"} and got.model == "claude-opus-5-5"


@pytest.mark.parametrize(
    "stdout,stderr,code,match",
    [
        ("not json", "boom", 1, "boom"),
        (json.dumps({"is_error": True, "subtype": "success", "result": "Overloaded"}), "", 1, "Overloaded"),
        (json.dumps({"is_error": True, "subtype": "error_during_execution", "errors": ["bad"]}),
         "", 1, "bad"),
        (json.dumps({"subtype": "success", "is_error": False, "result": "plain"}), "", 0, "no structured"),
    ],
)
def test_parse_envelope_errors(stdout, stderr, code, match):
    with pytest.raises(sm.SummaryError, match=match):
        sm.parse_envelope(proc(stdout, stderr, code))


def test_claude_command(tmp_path):
    plain = sm.claude_command(cfg_with(model="opus"), tmp_path, with_frames=False)
    assert plain[plain.index("--tools") + 1] == ""
    assert plain[plain.index("--model") + 1] == "opus" and "--bare" not in plain
    assert json.loads(plain[plain.index("--json-schema") + 1])["additionalProperties"] is False
    framed = sm.claude_command(cfg_with(), tmp_path, with_frames=True)
    assert framed[framed.index("--tools") + 1] == "Read"
    assert framed[framed.index("--add-dir") + 1] == str(tmp_path) and "--model" not in framed


@pytest.mark.parametrize("lang", ["de", "en"])
def test_end_to_end_fake_runner(tmp_path, lang):
    t = make_transcript(lang)
    session = make_session(tmp_path)
    calls = []

    def runner(cmd, stdin, cwd):
        calls.append((cmd, stdin, cwd))
        env = {"subtype": "success", "is_error": False, "structured_output": model_output(),
               "modelUsage": {"fake-model": {}}}
        return proc(json.dumps(env))

    out = sm.summarize_session(session, t, cfg_with(), keyframes=[], runner=runner)
    assert out == tmp_path / "summary.md" and calls[0][2] == tmp_path
    md = out.read_text()
    labels = sm.LABELS[lang]
    for key in ("summary", "key_points", "decisions", "action_items", "open_questions"):
        assert f"## {labels[key]}" in md
    assert "keine" in md if lang == "de" else "none" in md
    assert "| Release Notes schreiben | Jonas | nächste Woche Donnerstag → " in md
    assert "? ⚠️" in md and "[00:12]" in md and "[00:40] [30:00]" in md
    assert labels["warnings"] in md and labels["note"] in md
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["meta"]["model"] == "fake-model" and saved["meta"]["asr_model"] == "large-v3"
    assert saved["raw"]["action_items"][1]["owner"] == "Peter"
    assert saved["result"]["action_items"][1]["owner"] is None
    assert len(saved["warnings"]) == 5


def test_german_markdown_header(tmp_path):
    t = make_transcript()
    allowed, _ = sm.speaker_allow_list(t, cfg_with())
    res = sm.validate(model_output(), t, allowed, MONDAY, "de")
    md = sm.render_markdown(res, make_session(tmp_path), t, cfg_with(), "de")
    assert "- Datum: Montag, 14.09.2026" in md
    assert "- Uhrzeit: 10:00 (UTC+02:00)" in md
    assert "- Dauer: 01:30" in md
    assert "| Aufgabe | Verantwortlich | Frist | Beleg |" in md
    assert "Do 2026-09-24" in md
