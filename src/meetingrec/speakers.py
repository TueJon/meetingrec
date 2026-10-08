"""Speaker naming: the local voiceprint database and per-meeting identification."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

from .config import Config, data_dir
from .model import Transcript

VOICES_FILE = "voices.json"


def _unit(vector: list[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


@dataclass
class VoiceDB:
    """Stored voiceprints (biometric data: written with mode 600)."""

    entries: dict[str, dict] = field(default_factory=dict)
    path: Path = field(default_factory=lambda: data_dir() / VOICES_FILE)

    @classmethod
    def load(cls) -> VoiceDB:
        path = data_dir() / VOICES_FILE
        entries = json.loads(path.read_text()) if path.exists() else {}
        return cls(entries=entries, path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".voices-")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self.entries, fh, ensure_ascii=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def summary(self) -> list[tuple[str, int]]:
        return sorted((name, entry["count"]) for name, entry in self.entries.items())

    def forget(self, name: str) -> bool:
        if self.entries.pop(name, None) is None:
            return False
        self.save()
        return True

    def enroll(self, name: str, embedding: list[float], model: str) -> None:
        """Fold a sample into the running mean of L2-normalized vectors."""
        new = _unit(embedding)
        entry = self.entries.get(name)
        if entry is None or entry["model"] != model:
            self.entries[name] = {"embedding": new.tolist(), "count": 1, "model": model}
            return
        count = entry["count"]
        mean = _unit((np.asarray(entry["embedding"]) * count + new) / (count + 1))
        self.entries[name] = {"embedding": mean.tolist(), "count": count + 1, "model": model}

    def scores(self, embedding: list[float], model: str) -> dict[str, float]:
        """Cosine similarity to every voiceprint made with `model`."""
        probe = _unit(embedding)
        return {
            name: float(np.dot(probe, _unit(entry["embedding"])))
            for name, entry in self.entries.items()
            if entry["model"] == model
        }

    def match(
        self, embedding: list[float], model: str, threshold: float, margin: float
    ) -> tuple[str | None, float]:
        """Best voiceprint, or None when it is below `threshold` or too close to the runner-up."""
        ranked = sorted(self.scores(embedding, model).items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return None, 0.0
        name, best = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        if best < threshold or best - runner_up < margin:
            return None, best
        return name, best


def identify(transcript: Transcript, db: VoiceDB, cfg: Config) -> None:
    """Name "self" from the config and diarized clusters from stored voiceprints."""
    if "self" in transcript.speakers:
        info = transcript.speakers["self"]
        info.name = cfg.self_name or ("Ich" if transcript.language == "de" else "Me")
        info.source = "self"
    model = transcript.diarization_model
    if model is None:
        return
    for label, info in transcript.speakers.items():
        if label != "self" and info.source != "manual":
            info.name, info.source, info.similarity = None, None, None
    candidates = sorted(
        (
            (similarity, label, name)
            for label, info in transcript.speakers.items()
            if label != "self" and info.embedding and info.source != "manual"
            for name, similarity in [
                db.match(info.embedding, model, cfg.speakers.match_threshold, cfg.speakers.match_margin)
            ]
            if name is not None
        ),
        reverse=True,
    )
    taken = {info.name for info in transcript.speakers.values() if info.source == "manual"}
    for similarity, label, name in candidates:
        if name in taken:
            continue
        taken.add(name)
        info = transcript.speakers[label]
        info.name, info.source, info.similarity = name, "voiceprint", similarity


def assign_name(transcript: Transcript, label: str, name: str, db: VoiceDB | None) -> None:
    """Manually name a speaker and, when it has an embedding, remember the voice."""
    info = transcript.speakers.get(label)
    if info is None:
        valid = ", ".join(sorted(transcript.speakers)) or "none"
        raise click.ClickException(f"unknown speaker {label!r}; valid labels: {valid}")
    info.name, info.source, info.similarity = name, "manual", None
    if db is not None and info.embedding and transcript.diarization_model:
        db.enroll(name, info.embedding, transcript.diarization_model)
        db.save()
