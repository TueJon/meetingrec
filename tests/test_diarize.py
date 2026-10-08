import struct
import wave
from types import SimpleNamespace

import pytest

from meetingrec import diarize
from meetingrec.config import Config
from meetingrec.diarize import SpeakerTurn, assign_segment, assign_speakers, load_waveform
from meetingrec.model import Segment, SpeakerInfo, Transcript, Word
from meetingrec.session import Session


def words(*items: tuple[float, float, str]) -> list[Word]:
    return [Word(s, e, t) for s, e, t in items]


def seg(ws: list[Word], track="system") -> Segment:
    return Segment(ws[0].start, ws[-1].end, " ".join(w.text for w in ws), track, words=ws)


A, B = "SPEAKER_00", "SPEAKER_01"


def test_whole_segment_single_speaker_keeps_text() -> None:
    s = seg(words((0, 1, "Hallo"), (1, 2, "zusammen")))
    out = assign_speakers([s], [SpeakerTurn(0, 5, A)])
    assert [(o.speaker, o.text) for o in out] == [(A, "Hallo zusammen")]


def test_split_where_speaker_changes() -> None:
    tokens = ["ja", "genau", "und", "dann", "nein", "danke"]
    s = seg(words(*[(i, i + 1, t) for i, t in enumerate(tokens)]))
    out = assign_speakers([s], [SpeakerTurn(0, 3, A), SpeakerTurn(3, 6, B)])
    assert [(o.speaker, o.text, o.start, o.end) for o in out] == [
        (A, "ja genau und", 0, 3),
        (B, "dann nein danke", 3, 6),
    ]
    assert all(o.track == "system" for o in out)


def test_single_word_run_is_absorbed() -> None:
    s = seg(words((0, 1, "ja"), (1, 2, "genau"), (2, 3, "mhm"), (3, 4, "und"), (4, 5, "weiter")))
    out = assign_speakers([s], [SpeakerTurn(0, 2, A), SpeakerTurn(2, 3, B), SpeakerTurn(3, 5, A)])
    assert len(out) == 1 and out[0].speaker == A and out[0].text == s.text


def test_word_without_overlap_uses_nearest_turn() -> None:
    s = seg(words((10, 11, "hallo"), (11, 12, "du")))
    out = assign_speakers([s], [SpeakerTurn(0, 5, A), SpeakerTurn(8, 9, B)])
    assert out[0].speaker == B


def test_segment_without_words_uses_max_overlap() -> None:
    s = Segment(0, 10, "kein Wort-Timing", "system")
    out = assign_speakers([s], [SpeakerTurn(0, 3, A), SpeakerTurn(3, 10, B)])
    assert [o.speaker for o in out] == [B]


def test_no_turns_leaves_speaker_unset() -> None:
    s = seg(words((0, 1, "a"), (1, 2, "b")))
    assert assign_segment(s, [])[0].speaker is None


def test_load_waveform_reads_pcm16(tmp_path) -> None:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(struct.pack("<4h", 0, 16384, -16384, 32767))
    pytest.importorskip("torch")
    tensor, rate = load_waveform(path)
    assert rate == 16000 and tuple(tensor.shape) == (1, 4)
    assert tensor[0, 1].item() == pytest.approx(0.5)


class FakeAnnotation:
    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, yield_label: bool = False):
        for start, end, label in self._turns:
            yield SimpleNamespace(start=start, end=end), None, label

    def labels(self) -> list[str]:
        return sorted({label for *_, label in self._turns})


class FakePipeline:
    def __call__(self, audio: dict, **kwargs):
        assert audio["sample_rate"] == 16000 and kwargs == {"min_speakers": None, "max_speakers": None}
        annotation = FakeAnnotation([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")])
        return SimpleNamespace(
            speaker_diarization=annotation,
            exclusive_speaker_diarization=annotation,
            speaker_embeddings=[[1.0, 0.0], [0.0, float("nan")]],
        )


def test_diarize_session_labels_system_segments_only(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    meeting = tmp_path / "2026-08-15_095058_t"
    meeting.mkdir()
    with wave.open(str(meeting / "audio.wav"), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"\0\0" * 16000)
    session = Session.load(meeting)
    mic = Segment(1, 2, "ich", "mic", speaker="self")
    letters = words(*[(i, i + 1, t) for i, t in enumerate("abcdef")])
    remote = Segment(0, 6, "a b c d e f", "mixed", words=letters)
    transcript = Transcript("de", [mic, remote], speakers={"self": SpeakerInfo("self", source="self")})
    monkeypatch.setattr(diarize, "_load_pipeline", lambda cfg: FakePipeline())

    diarize.diarize_session(session, transcript, Config())

    assert [(s.speaker, s.text) for s in transcript.segments if s.track == "mixed"] == [
        ("SPEAKER_00", "a b c"),
        ("SPEAKER_01", "d e f"),
    ]
    assert transcript.segments[1].speaker == "self"  # untouched, still ordered by time
    assert transcript.diarization_model == Config().diarization.model
    assert transcript.speakers["SPEAKER_00"].embedding == [1.0, 0.0]
    assert transcript.speakers["SPEAKER_01"].embedding is None  # non-finite vector
    assert transcript.speakers["self"].source == "self"


def test_missing_token_warns_and_leaves_speakers_unset(capsys, monkeypatch) -> None:
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    pytest.importorskip("pyannote.audio")
    assert diarize._load_pipeline(Config()) is None
    err = capsys.readouterr().err
    assert "hf auth login" in err and "uv sync --extra diarize" in err
