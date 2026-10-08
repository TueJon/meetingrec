"""Speaker diarization with pyannote.audio: who spoke when on the remote-audio track."""

from __future__ import annotations

import time
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

from .config import Config
from .model import Segment, SpeakerInfo, TrackRole, Transcript, Word
from .session import Session

# A change of speaker inside a segment needs at least this many words to split it off.
MIN_RUN_WORDS = 2


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


# ---- assigning turns to transcript segments --------------------------------


def _overlap(start: float, end: float, turn: SpeakerTurn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


def _distance(start: float, end: float, turn: SpeakerTurn) -> float:
    return max(turn.start - end, start - turn.end, 0.0)


def _speaker_for(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Speaker with the most overlap, else the nearest turn; None without any turn."""
    if not turns:
        return None
    best = max(turns, key=lambda t: _overlap(start, end, t))
    if _overlap(start, end, best) > 0:
        return best.speaker
    return min(turns, key=lambda t: _distance(start, end, t)).speaker


def _runs(words: list[Word], turns: list[SpeakerTurn]) -> list[tuple[str | None, list[Word]]]:
    runs: list[tuple[str | None, list[Word]]] = []
    for word in words:
        speaker = _speaker_for(word.start, word.end, turns)
        if runs and runs[-1][0] == speaker:
            runs[-1][1].append(word)
        else:
            runs.append((speaker, [word]))
    return runs


def _coalesce(runs: list[tuple[str | None, list[Word]]]) -> list[tuple[str | None, list[Word]]]:
    merged: list[tuple[str | None, list[Word]]] = []
    for speaker, words in runs:
        if merged and merged[-1][0] == speaker:
            merged[-1][1].extend(words)
        else:
            merged.append((speaker, words))
    return merged


def _absorb_short_runs(runs: list[tuple[str | None, list[Word]]]) -> list[tuple[str | None, list[Word]]]:
    """Merge runs shorter than MIN_RUN_WORDS into their longer neighbour until stable."""
    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda i: len(runs[i][1]))
        if len(runs[shortest][1]) >= MIN_RUN_WORDS:
            break
        neighbours = [i for i in (shortest - 1, shortest + 1) if 0 <= i < len(runs)]
        target = max(neighbours, key=lambda i: len(runs[i][1]))
        first, second = sorted((shortest, target))
        merged = (runs[target][0], runs[first][1] + runs[second][1])
        runs[first : second + 1] = [merged]
        runs = _coalesce(runs)
    return runs


def assign_segment(seg: Segment, turns: list[SpeakerTurn]) -> list[Segment]:
    """Label one segment, splitting it where the speaker changes between words."""
    if not seg.words:
        seg.speaker = _speaker_for(seg.start, seg.end, turns)
        return [seg]
    runs = _absorb_short_runs(_runs(seg.words, turns))
    if len(runs) == 1:
        seg.speaker = runs[0][0]
        return [seg]
    return [
        Segment(
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(w.text for w in words),
            track=seg.track,
            speaker=speaker,
            words=words,
        )
        for speaker, words in runs
    ]


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Speaker-labelled copy of `segments`; segments may be split at speaker changes."""
    return [piece for seg in segments for piece in assign_segment(seg, turns)]


# ---- running pyannote -----------------------------------------------------


def load_waveform(wav: Path):
    """16-bit PCM WAV as a (channels=1, samples) float tensor plus its sample rate."""
    import torch

    with wave.open(str(wav)) as fh:
        if fh.getsampwidth() != 2:
            raise ValueError(f"{wav} is not 16-bit PCM")
        channels, rate = fh.getnchannels(), fh.getframerate()
        samples = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    mono = samples.reshape(-1, channels).mean(axis=1) / 32768.0
    return torch.from_numpy(mono.astype(np.float32))[None, :], rate


def _warn_unavailable(model: str, reason: str) -> None:
    click.echo(
        f"[diarize] skipped: {reason}\n"
        "          Remote speakers stay unlabelled. To enable diarization:\n"
        "            1. uv sync --extra diarize\n"
        f"            2. accept the model conditions at https://huggingface.co/{model}\n"
        "            3. hf auth login",
        err=True,
    )


def _load_pipeline(cfg: Config):
    """The pyannote pipeline on its device, or None after printing why it is unavailable."""
    model = cfg.diarization.model
    try:
        with warnings.catch_warnings():
            # pyannote warns that torchcodec cannot decode files; we pass audio in memory.
            warnings.filterwarnings("ignore", message="(?s).*torchcodec.*")
            import torch
            from huggingface_hub import get_token
            from pyannote.audio import Pipeline
    except ImportError as exc:
        _warn_unavailable(model, f"pyannote.audio is not installed ({exc.name})")
        return None
    token = get_token()
    if token is None:
        _warn_unavailable(model, "no Hugging Face token found")
        return None
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except Exception as exc:  # gated/unknown repo, bad token, offline
        _warn_unavailable(model, f"could not load {model}: {type(exc).__name__}")
        return None
    if pipeline is None:
        _warn_unavailable(model, f"{model} refused access (conditions not accepted?)")
        return None
    device = cfg.diarization.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"[diarize] model={model} device={device}")
    return pipeline.to(torch.device(device))


def _turns(annotation) -> list[SpeakerTurn]:
    return [SpeakerTurn(t.start, t.end, label) for t, _, label in annotation.itertracks(yield_label=True)]


def diarization_track(session: Session) -> TrackRole | None:
    """The remote participants' audio: the system track, else a legacy mixed track."""
    for role in ("system", "mixed"):
        if session.track(role):
            return role
    return None


def diarize_session(session: Session, transcript: Transcript, cfg: Config) -> None:
    role = diarization_track(session)
    if role is None:
        return
    pipeline = _load_pipeline(cfg)
    if pipeline is None:
        return
    wav, offset = session.track_audio(role)
    waveform, rate = load_waveform(wav)
    started = time.monotonic()
    output = pipeline(
        {"waveform": waveform, "sample_rate": rate},
        min_speakers=cfg.diarization.min_speakers,
        max_speakers=cfg.diarization.max_speakers,
    )
    turns = [
        SpeakerTurn(t.start + offset, t.end + offset, t.speaker)
        for t in _turns(output.exclusive_speaker_diarization)
    ]
    embeddings = dict(zip(output.speaker_diarization.labels(), output.speaker_embeddings, strict=False))

    others = [s for s in transcript.segments if s.track != role]
    assigned = assign_speakers([s for s in transcript.segments if s.track == role], turns)
    transcript.segments = sorted(others + assigned, key=lambda s: s.start)
    transcript.diarization_model = cfg.diarization.model
    for label in sorted({s.speaker for s in assigned if s.speaker}):
        transcript.speakers[label] = SpeakerInfo(label=label, embedding=_embedding(embeddings.get(label)))
    click.echo(
        f"[diarize] {len(set(s.speaker for s in assigned if s.speaker))} speaker(s) on the {role} track "
        f"in {time.monotonic() - started:.0f}s"
    )


def _embedding(vector) -> list[float] | None:
    if vector is None or not np.all(np.isfinite(vector)):
        return None
    return [float(x) for x in vector]
