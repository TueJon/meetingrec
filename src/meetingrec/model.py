"""Core data model shared by every pipeline stage.

Times are seconds on the meeting timeline: 0.0 is the moment the recording
pipeline started, for every track and for the screen video.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

TrackRole = Literal["mic", "system", "mixed"]


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    track: TrackRole
    # Raw diarization label ("SPEAKER_01"), "self" for the mic track, or None.
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)


@dataclass
class SpeakerInfo:
    """One speaker cluster of this meeting."""

    label: str
    name: str | None = None
    # How `name` was assigned: "self", "voiceprint", "manual", or None.
    source: str | None = None
    similarity: float | None = None
    embedding: list[float] | None = None


@dataclass
class Transcript:
    language: str
    segments: list[Segment]
    speakers: dict[str, SpeakerInfo] = field(default_factory=dict)
    asr_model: str = ""
    diarization_model: str | None = None

    def display_name(self, label: str | None) -> str:
        if label is None:
            return "?"
        info = self.speakers.get(label)
        return info.name if info and info.name else label

    def to_dict(self) -> dict:
        return {"version": 2, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> Transcript:
        return cls(
            language=data["language"],
            segments=[
                Segment(**{**s, "words": [Word(**w) for w in s.get("words", [])]}) for s in data["segments"]
            ],
            speakers={k: SpeakerInfo(**v) for k, v in data.get("speakers", {}).items()},
            asr_model=data.get("asr_model", ""),
            diarization_model=data.get("diarization_model"),
        )


@dataclass
class Keyframe:
    """A screen frame worth showing to the summarizer."""

    time: float
    path: str  # relative to the meeting directory
