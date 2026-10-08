from meetingrec.model import Segment
from meetingrec.transcribe import drop_echo, resolve_model


def seg(start: float, end: float, text: str, track="mic") -> Segment:
    return Segment(start, end, text, track, speaker="self" if track == "mic" else None)


def test_model_aliases_and_passthrough() -> None:
    assert resolve_model("large-v3") == "Systran/faster-whisper-large-v3"
    assert resolve_model("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert resolve_model("someone/custom-ct2") == "someone/custom-ct2"
    assert resolve_model("/models/local") == "/models/local"


def test_echo_of_system_audio_is_dropped() -> None:
    system = [seg(10.0, 14.0, "Wir sollten das Angebot bis Freitag schicken", "system")]
    mic = [seg(10.4, 14.3, "wir sollten das angebot bis freitag schicken.")]
    assert drop_echo(mic, system) == []


def test_own_speech_is_kept_even_while_remote_talks() -> None:
    system = [seg(10.0, 14.0, "Wir sollten das Angebot bis Freitag schicken", "system")]
    mic = [seg(11.0, 13.0, "Ja, das passt mir gut")]
    assert drop_echo(mic, system) == mic


def test_similar_text_outside_the_time_window_is_kept() -> None:
    system = [seg(10.0, 14.0, "Wir sollten das Angebot bis Freitag schicken", "system")]
    mic = [seg(60.0, 64.0, "Wir sollten das Angebot bis Freitag schicken")]
    assert drop_echo(mic, system) == mic


def test_empty_system_keeps_everything() -> None:
    mic = [seg(0, 2, "Hallo")]
    assert drop_echo(mic, []) == mic
