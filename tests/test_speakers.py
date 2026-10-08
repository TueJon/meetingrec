import json
import stat

import click
import pytest

from meetingrec.config import Config
from meetingrec.model import SpeakerInfo, Transcript
from meetingrec.speakers import VoiceDB, assign_name, identify

MODEL = "pyannote/speaker-diarization-community-1"


@pytest.fixture(autouse=True)
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


def vec(*xs: float) -> list[float]:
    return list(xs) + [0.0] * (8 - len(xs))


def transcript(**speakers: SpeakerInfo) -> Transcript:
    return Transcript(language="de", segments=[], speakers=speakers, diarization_model=MODEL)


def test_enroll_is_running_mean_of_normalized_vectors() -> None:
    db = VoiceDB.load()
    db.enroll("Anna", vec(10, 0), MODEL)
    db.enroll("Anna", vec(0, 2), MODEL)
    entry = db.entries["Anna"]
    assert entry["count"] == 2
    assert entry["embedding"][0] == pytest.approx(2**-0.5)
    assert entry["embedding"][1] == pytest.approx(2**-0.5)


def test_match_threshold_and_margin() -> None:
    db = VoiceDB.load()
    db.enroll("Anna", vec(1, 0), MODEL)
    db.enroll("Ben", vec(0, 1), MODEL)
    assert db.match(vec(1, 0.1), MODEL, 0.55, 0.08)[0] == "Anna"
    assert db.match(vec(1, 1), MODEL, 0.55, 0.08)[0] is None  # tie: margin fails
    assert db.match(vec(0, 0, 1), MODEL, 0.55, 0.08)[0] is None  # below threshold


def test_match_ignores_other_models_and_empty_db() -> None:
    db = VoiceDB.load()
    assert db.match(vec(1), MODEL, 0.5, 0.0) == (None, 0.0)
    db.enroll("Anna", vec(1), "other-model")
    assert db.match(vec(1), MODEL, 0.5, 0.0) == (None, 0.0)


def test_persistence_forget_and_private_file(tmp_path) -> None:
    db = VoiceDB.load()
    db.enroll("Anna", vec(1), MODEL)
    db.enroll("Anna", vec(1), MODEL)
    db.save()
    path = tmp_path / "meetingrec" / "voices.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["Anna"]["count"] == 2
    assert VoiceDB.load().summary() == [("Anna", 2)]
    assert VoiceDB.load().forget("Anna") is True
    assert VoiceDB.load().forget("Anna") is False
    assert VoiceDB.load().summary() == []


def test_identify_names_self_by_language_and_config() -> None:
    t = transcript(self=SpeakerInfo("self", source="self"))
    identify(t, VoiceDB.load(), Config())
    assert t.speakers["self"].name == "Ich"
    t.language = "en"
    identify(t, VoiceDB.load(), Config())
    assert t.speakers["self"].name == "Me"
    identify(t, VoiceDB.load(), Config(self_name="Jonas"))
    assert t.speakers["self"].name == "Jonas"


def test_identify_matches_voiceprints_and_keeps_names_unique() -> None:
    db = VoiceDB.load()
    db.enroll("Anna", vec(1, 0), MODEL)
    t = transcript(
        SPEAKER_00=SpeakerInfo("SPEAKER_00", embedding=vec(1, 0.3)),
        SPEAKER_01=SpeakerInfo("SPEAKER_01", embedding=vec(1, 0.05)),
        SPEAKER_02=SpeakerInfo("SPEAKER_02", embedding=vec(0, 0, 1)),
        SPEAKER_03=SpeakerInfo("SPEAKER_03"),
    )
    identify(t, db, Config())
    assert t.speakers["SPEAKER_01"].name == "Anna"
    assert t.speakers["SPEAKER_01"].source == "voiceprint"
    assert t.speakers["SPEAKER_01"].similarity > 0.99
    assert t.speakers["SPEAKER_00"].name is None
    assert t.speakers["SPEAKER_02"].name is None
    assert t.speakers["SPEAKER_03"].name is None


def test_identify_respects_manual_names() -> None:
    db = VoiceDB.load()
    db.enroll("Anna", vec(1, 0), MODEL)
    t = transcript(
        SPEAKER_00=SpeakerInfo("SPEAKER_00", name="Anna", source="manual", embedding=vec(0, 1)),
        SPEAKER_01=SpeakerInfo("SPEAKER_01", embedding=vec(1, 0)),
    )
    identify(t, db, Config())
    assert t.speakers["SPEAKER_00"].source == "manual"
    assert t.speakers["SPEAKER_01"].name is None


def test_identify_clears_stale_voiceprint_names() -> None:
    t = transcript(SPEAKER_00=SpeakerInfo("SPEAKER_00", name="Old", source="voiceprint", embedding=vec(1)))
    identify(t, VoiceDB.load(), Config())
    assert t.speakers["SPEAKER_00"].name is None


def test_assign_name_unknown_label_lists_valid_ones() -> None:
    t = transcript(SPEAKER_00=SpeakerInfo("SPEAKER_00"))
    with pytest.raises(click.ClickException, match="SPEAKER_00"):
        assign_name(t, "SPEAKER_09", "Anna", None)


def test_assign_name_enrolls_when_embedding_present() -> None:
    t = transcript(
        SPEAKER_00=SpeakerInfo("SPEAKER_00", embedding=vec(1)),
        SPEAKER_01=SpeakerInfo("SPEAKER_01"),
    )
    assign_name(t, "SPEAKER_01", "Ben", VoiceDB.load())
    assert t.speakers["SPEAKER_01"].source == "manual"
    assert VoiceDB.load().summary() == []
    assign_name(t, "SPEAKER_00", "Anna", VoiceDB.load())
    assert VoiceDB.load().summary() == [("Anna", 1)]
    assign_name(t, "SPEAKER_00", "Anna2", None)
    assert VoiceDB.load().summary() == [("Anna", 1)]
