"""Transcript outputs: transcript.{json,md,txt,srt,vtt}."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .model import Segment, Transcript
from .session import Session

# Consecutive segments of one speaker closer than this merge into one turn.
TURN_GAP = 2.0


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str


def format_ts(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _cue_ts(seconds: float, sep: str) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def speaker_name(transcript: Transcript, seg: Segment) -> str:
    if seg.speaker is None:
        return "Remote" if seg.track == "system" else "?"
    return transcript.display_name(seg.speaker)


def turns(transcript: Transcript) -> list[Turn]:
    out: list[Turn] = []
    for seg in sorted(transcript.segments, key=lambda s: s.start):
        name = speaker_name(transcript, seg)
        text = seg.text.strip()
        if not text:
            continue
        if out and out[-1].speaker == name and seg.start - out[-1].end < TURN_GAP:
            out[-1].text += " " + text
            out[-1].end = max(out[-1].end, seg.end)
        else:
            out.append(Turn(seg.start, seg.end, name, text))
    return out


def transcript_lines(transcript: Transcript) -> list[str]:
    """One "[mm:ss] Name: text" line per speaker turn — the summarizer's input format."""
    return [f"[{format_ts(t.start)}] {t.speaker}: {t.text}" for t in turns(transcript)]


def participants(transcript: Transcript) -> list[str]:
    """Speakers in order of first appearance; unattributed speech ("?") is not a participant."""
    seen: dict[str, None] = {}
    for t in turns(transcript):
        if t.speaker != "?":
            seen.setdefault(t.speaker, None)
    return list(seen)


def write_transcript(session: Session, transcript: Transcript) -> None:
    d = session.dir
    (d / "transcript.json").write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1) + "\n")

    all_turns = turns(transcript)
    meta = session.meta
    duration = max((s.end for s in transcript.segments), default=0.0)
    de = transcript.language == "de"
    labels = ("Beginn", "Dauer", "Sprecher", "Modelle") if de else ("Start", "Duration", "Speakers", "Models")
    models = ", ".join(m for m in (transcript.asr_model, transcript.diarization_model) if m)
    md = [
        f"# {meta.title}",
        "",
        f"- {labels[0]}: {meta.started:%Y-%m-%d %H:%M}",
        f"- {labels[1]}: {format_ts(duration)}",
        f"- {labels[2]}: {', '.join(participants(transcript)) or '—'}",
        f"- {labels[3]}: {models}",
        "",
    ]
    md += [f"**[{format_ts(t.start)}] {t.speaker}:** {t.text}\n" for t in all_turns]
    (d / "transcript.md").write_text("\n".join(md))
    (d / "transcript.txt").write_text("\n".join(transcript_lines(transcript)) + "\n")

    segs = [s for s in sorted(transcript.segments, key=lambda s: s.start) if s.text.strip()]
    srt, vtt = [], ["WEBVTT", ""]
    for i, s in enumerate(segs, 1):
        name = speaker_name(transcript, s)
        srt += [str(i), f"{_cue_ts(s.start, ',')} --> {_cue_ts(s.end, ',')}", f"{name}: {s.text.strip()}", ""]
        vtt += [f"{_cue_ts(s.start, '.')} --> {_cue_ts(s.end, '.')}", f"<v {name}>{s.text.strip()}", ""]
    (d / "transcript.srt").write_text("\n".join(srt))
    (d / "transcript.vtt").write_text("\n".join(vtt))
