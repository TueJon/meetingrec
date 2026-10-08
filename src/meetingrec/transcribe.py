"""Speech recognition with faster-whisper: one pass per recorded track."""

from __future__ import annotations

import ctypes
import difflib
import importlib.util
import re
import time
import wave
from pathlib import Path

import click

from .config import Config
from .model import Segment, SpeakerInfo, TrackRole, Transcript, Word
from .session import Session

MODEL_ALIASES = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# A mic segment this similar to the system audio at the same moment is loudspeaker bleed.
ECHO_SIMILARITY = 0.6
ECHO_WINDOW = 1.5

_CUDA_LIBS = (("cublas", "libcublas.so.12"),)


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def _preload_cuda_libs() -> None:
    """Load the pip-installed cuBLAS so CTranslate2 finds it without LD_LIBRARY_PATH."""
    spec = importlib.util.find_spec("nvidia")
    for base in (spec.submodule_search_locations or []) if spec else []:
        for package, soname in _CUDA_LIBS:
            lib = Path(base) / package / "lib" / soname
            if lib.exists():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def pick_device(cfg: Config) -> str:
    import ctranslate2

    if cfg.asr.device != "auto":
        return cfg.asr.device
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def pick_compute_type(cfg: Config, device: str) -> str:
    import ctranslate2

    if cfg.asr.compute_type != "auto":
        return cfg.asr.compute_type
    # Tensor-core GPUs run int8 weights with float16 math fastest; older ones
    # (e.g. Pascal) offer no float16 kernels, and plain int8 is their fastest type.
    if device == "cuda" and "int8_float16" in ctranslate2.get_supported_compute_types("cuda"):
        return "int8_float16"
    return "int8"


def load_model(cfg: Config, model_id: str):
    from faster_whisper import WhisperModel

    device = pick_device(cfg)
    if device == "cuda":
        _preload_cuda_libs()
    compute_type = pick_compute_type(cfg, device)
    click.echo(f"[asr] model={model_id} device={device} compute_type={compute_type}")
    return WhisperModel(model_id, device=device, compute_type=compute_type)


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as fh:
        return fh.getnframes() / fh.getframerate()


def detect_language(model, wav: Path) -> str:
    from faster_whisper import decode_audio

    language, probability, _ = model.detect_language(decode_audio(str(wav)), vad_filter=True)
    click.echo(f"[asr] detected language {language} ({probability:.0%})")
    return language


def decode_track(
    model, wav: Path, offset: float, role: TrackRole, language: str, cfg: Config
) -> list[Segment]:
    options: dict = {
        "language": language,
        "beam_size": cfg.asr.beam_size,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    if cfg.glossary:
        options["hotwords"] = ", ".join(cfg.glossary)
    raw, _ = model.transcribe(str(wav), **options)
    speaker = "self" if role == "mic" else None
    return [
        Segment(
            start=seg.start + offset,
            end=seg.end + offset,
            text=seg.text.strip(),
            track=role,
            speaker=speaker,
            words=[
                Word(w.start + offset, w.end + offset, w.word.strip(), w.probability) for w in seg.words or []
            ],
        )
        for seg in raw
        if seg.text.strip()
    ]


def _word_list(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def drop_echo(mic: list[Segment], system: list[Segment]) -> list[Segment]:
    """Remove mic segments that merely repeat the system audio (speaker-to-mic bleed)."""
    kept = []
    for seg in mic:
        overlapping = [
            s.text for s in system if s.start < seg.end + ECHO_WINDOW and s.end > seg.start - ECHO_WINDOW
        ]
        mine = _word_list(seg.text)
        theirs = _word_list(" ".join(overlapping))
        if mine and theirs and difflib.SequenceMatcher(None, mine, theirs).ratio() >= ECHO_SIMILARITY:
            continue
        kept.append(seg)
    return kept


def transcribe_session(session: Session, cfg: Config, *, fast: bool = False) -> Transcript:
    model_id = resolve_model(cfg.asr.fast_model if fast else cfg.asr.model)
    model = load_model(cfg, model_id)
    tracks = {t.role: session.track_audio(t.role) for t in session.meta.tracks}
    durations = {role: wav_duration(wav) for role, (wav, _) in tracks.items()}

    language = cfg.language
    if language == "auto":
        language = detect_language(model, tracks[max(durations, key=durations.get)][0])

    by_role: dict[TrackRole, list[Segment]] = {}
    for role, (wav, offset) in tracks.items():
        started = time.monotonic()
        by_role[role] = decode_track(model, wav, offset, role, language, cfg)
        elapsed = time.monotonic() - started
        click.echo(
            f"[asr] {role}: {durations[role]:.0f}s audio in {elapsed:.0f}s "
            f"(real-time factor {elapsed / max(durations[role], 1e-9):.2f})"
        )

    if "mic" in by_role and "system" in by_role:
        before = len(by_role["mic"])
        by_role["mic"] = drop_echo(by_role["mic"], by_role["system"])
        if before != len(by_role["mic"]):
            dropped = before - len(by_role["mic"])
            click.echo(f"[asr] dropped {dropped} mic segment(s) that echo the system audio")

    segments = sorted((s for segs in by_role.values() for s in segs), key=lambda s: s.start)
    speakers = {"self": SpeakerInfo(label="self", source="self")} if "mic" in by_role else {}
    return Transcript(language=language, segments=segments, speakers=speakers, asr_model=model_id)
