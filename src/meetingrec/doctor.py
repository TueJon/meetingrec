"""`meetingrec doctor`: check everything a recording and its processing need."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .capture import portal, recorder
from .config import Config, config_dir

Level = Literal["ok", "warn", "fail"]
_MARKS: dict[str, str] = {"ok": "✓", "warn": "⚠", "fail": "✗"}

AUDIO_ELEMENTS = ("pulsesrc", "flacenc", "matroskamux", "audioconvert", "audioresample")
SCREEN_ELEMENTS = ("pipewiresrc", "videorate", "h264parse")
VIDEO_ENCODERS = ("nvh264enc", "x264enc")
ASR_REPOS = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


@dataclass
class Result:
    level: Level
    text: str

    def __str__(self) -> str:
        return f"{_MARKS[self.level]} {self.text}"


def _ok(text: str) -> Result:
    return Result("ok", text)


def _warn(text: str) -> Result:
    return Result("warn", text)


def _fail(text: str) -> Result:
    return Result("fail", text)


def _run(argv: list[str], timeout: float = 5) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def asr_repo(model: str) -> str:
    return ASR_REPOS.get(model, model)


# ---- capture ----------------------------------------------------------------


def _has_element(name: str) -> bool:
    proc = _run(["gst-inspect-1.0", "--exists", name])
    return proc is not None and proc.returncode == 0


def check_gstreamer() -> list[Result]:
    if not shutil.which("gst-launch-1.0") or not shutil.which("gst-inspect-1.0"):
        return [_fail("GStreamer: gst-launch-1.0 not found (install gstreamer1.0-tools)")]
    results = [_ok("gst-launch-1.0")]
    missing = [e for e in AUDIO_ELEMENTS if not _has_element(e)]
    if missing:
        results.append(_fail(f"GStreamer audio elements missing: {', '.join(missing)}"))
    else:
        results.append(_ok("GStreamer audio elements"))
    missing = [e for e in SCREEN_ELEMENTS if not _has_element(e)]
    encoders = [e for e in VIDEO_ENCODERS if _has_element(e)]
    if missing or not encoders:
        need = missing + ([] if encoders else ["a video encoder (nvh264enc or x264enc)"])
        results.append(_warn(f"screen capture unavailable, missing: {', '.join(need)}"))
    else:
        results.append(_ok(f"GStreamer screen elements (encoders: {', '.join(encoders)})"))
    return results


def check_audio_sources(cfg: Config) -> Result:
    try:
        mic, system = recorder.resolve_sources(cfg, None, None)
    except RuntimeError as exc:
        return _fail(str(exc).replace("\n", " "))
    return _ok(f"audio sources: mic {mic}, system {system}")


def check_portal() -> list[Result]:
    try:
        version = portal.screencast_version()
    except Exception as exc:  # noqa: BLE001 - any D-Bus failure just means "no portal"
        return [_warn(f"ScreenCast portal unreachable ({exc}); --screen will not work")]
    stored = portal.restore_token_path().exists()
    note = "monitor choice remembered" if stored else "no monitor remembered yet"
    return [_ok(f"ScreenCast portal v{version}, {note}")]


# ---- tools ------------------------------------------------------------------


def check_ffmpeg() -> list[Result]:
    return [
        _ok(name) if shutil.which(name) else _fail(f"{name} not found (install ffmpeg)")
        for name in ("ffmpeg", "ffprobe")
    ]


def check_claude() -> Result:
    proc = _run(["claude", "--version"])
    if proc is None or proc.returncode != 0:
        return _warn("claude CLI not found; summaries will not work")
    return _ok(f"claude {proc.stdout.strip()}")


# ---- models -----------------------------------------------------------------


def check_cuda() -> Result:
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001 - a broken CUDA install must not crash the doctor
        return _warn(f"CUDA check failed ({exc}); transcription will run on CPU")
    if count:
        return _ok(f"CUDA devices: {count}")
    return _warn("no CUDA device; transcription will run on CPU (slow)")


def _model_cached(model: str) -> bool:
    if Path(model).expanduser().exists():
        return True
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(asr_repo(model), local_files_only=True)
    except LocalEntryNotFoundError:
        return False
    return True


def check_asr_model(cfg: Config) -> list[Result]:
    results = []
    for model in dict.fromkeys([cfg.asr.model, cfg.asr.fast_model]):
        if _model_cached(model):
            results.append(_ok(f"ASR model {model} cached"))
        else:
            results.append(_warn(f"ASR model {model} not cached; downloads on first use ({asr_repo(model)})"))
    return results


def check_diarization(cfg: Config) -> list[Result]:
    if not cfg.diarization.enabled:
        return [_ok("diarization disabled in config")]
    if importlib.util.find_spec("pyannote") is None:
        return [
            _warn("pyannote.audio not installed; speakers will not be separated (uv sync --extra diarize)")
        ]
    results = [_ok("pyannote.audio installed")]
    from huggingface_hub import auth_check, get_token
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    model = cfg.diarization.model
    hint = f"accept the conditions on https://huggingface.co/{model} and run `hf auth login`"
    if not get_token():
        return [*results, _warn(f"no Hugging Face token; {hint}")]
    try:
        auth_check(model)
    except (GatedRepoError, RepositoryNotFoundError):
        results.append(_warn(f"no access to {model}; {hint}"))
    except (HfHubHTTPError, OSError) as exc:
        results.append(_warn(f"could not verify access to {model} (offline?): {exc}"))
    else:
        results.append(_ok(f"access to {model}"))
    return results


# ---- config -----------------------------------------------------------------


def _writable(path: Path) -> bool:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return os.access(probe, os.W_OK)


def check_config(cfg: Config) -> list[Result]:
    where = str(cfg.source) if cfg.source else f"{config_dir() / 'config.toml'} (absent, defaults in use)"
    results = [_ok(f"config: {where}, glossary {len(cfg.glossary)} term(s)")]
    if _writable(cfg.out_root):
        results.append(_ok(f"out_root writable: {cfg.out_root}"))
    else:
        results.append(_fail(f"out_root not writable: {cfg.out_root}"))
    return results


# ---- entry point ------------------------------------------------------------


def run_doctor(cfg: Config) -> int:
    results: list[Result] = [
        *check_gstreamer(),
        check_audio_sources(cfg),
        *check_portal(),
        *check_ffmpeg(),
        check_cuda(),
        *check_asr_model(cfg),
        *check_diarization(cfg),
        check_claude(),
        *check_config(cfg),
    ]
    for result in results:
        print(result)
    return 1 if any(r.level == "fail" for r in results) else 0
