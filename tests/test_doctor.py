import importlib.util
import subprocess

import pytest

from meetingrec import doctor
from meetingrec.config import Config


def _proc(code: int, out: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, "")


def test_asr_repo_mapping():
    assert doctor.asr_repo("large-v3") == "Systran/faster-whisper-large-v3"
    assert doctor.asr_repo("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert doctor.asr_repo("org/custom") == "org/custom"


def test_result_rendering():
    assert str(doctor.Result("ok", "x")) == "✓ x"
    assert str(doctor.Result("warn", "x")) == "⚠ x"
    assert str(doctor.Result("fail", "x")) == "✗ x"


def test_gstreamer_missing_is_required_failure(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    assert [r.level for r in doctor.check_gstreamer()] == ["fail"]


def test_missing_screen_elements_only_warn(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(doctor, "_has_element", lambda name: name != "pipewiresrc")
    levels = [r.level for r in doctor.check_gstreamer()]
    assert levels == ["ok", "ok", "warn"]


def test_missing_audio_element_fails(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(doctor, "_has_element", lambda name: name != "flacenc")
    assert doctor.check_gstreamer()[1].level == "fail"


def test_audio_source_error_is_failure(monkeypatch):
    def boom(*_args):
        raise RuntimeError("mic source 'x' not found. Available sources:\n  a")

    monkeypatch.setattr(doctor.recorder, "resolve_sources", boom)
    result = doctor.check_audio_sources(Config())
    assert result.level == "fail"
    assert "\n" not in result.text


def test_portal_unreachable_only_warns(monkeypatch):
    def boom():
        raise OSError("no bus")

    monkeypatch.setattr(doctor.portal, "screencast_version", boom)
    assert doctor.check_portal()[0].level == "warn"


def test_claude_missing_warns(monkeypatch):
    monkeypatch.setattr(doctor, "_run", lambda *_a, **_k: None)
    assert doctor.check_claude().level == "warn"


def test_out_root_writability_uses_nearest_existing_parent(tmp_path):
    assert doctor._writable(tmp_path / "a" / "b")
    tmp_path.chmod(0o500)
    try:
        assert not doctor._writable(tmp_path / "a" / "b")
    finally:
        tmp_path.chmod(0o700)


def test_diarization_disabled():
    cfg = Config()
    cfg.diarization.enabled = False
    assert [r.level for r in doctor.check_diarization(cfg)] == ["ok"]


def test_diarization_without_pyannote_hints_extra(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    (result,) = doctor.check_diarization(Config())
    assert result.level == "warn"
    assert "uv sync --extra diarize" in result.text


def test_diarization_without_token_hints_login(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    results = doctor.check_diarization(Config())
    assert [r.level for r in results] == ["ok", "warn"]
    assert "hf auth login" in results[1].text


def test_asr_model_cache_miss_warns(monkeypatch):
    monkeypatch.setattr(doctor, "_model_cached", lambda _m: False)
    assert {r.level for r in doctor.check_asr_model(Config())} == {"warn"}


@pytest.mark.parametrize("level,code", [("ok", 0), ("warn", 0), ("fail", 1)])
def test_exit_code(monkeypatch, capsys, level, code):
    for name in (
        "check_gstreamer",
        "check_portal",
        "check_ffmpeg",
        "check_asr_model",
        "check_diarization",
        "check_config",
    ):
        monkeypatch.setattr(doctor, name, lambda *_a: [])
    monkeypatch.setattr(doctor, "check_audio_sources", lambda _c: doctor.Result(level, "audio"))
    monkeypatch.setattr(doctor, "check_cuda", lambda: doctor.Result("ok", "cuda"))
    monkeypatch.setattr(doctor, "check_claude", lambda: doctor.Result("ok", "claude"))
    assert doctor.run_doctor(Config()) == code
    assert "audio" in capsys.readouterr().out
