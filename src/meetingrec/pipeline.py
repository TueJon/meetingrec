"""Post-recording processing: transcribe -> diarize -> identify -> render -> summarize."""

from __future__ import annotations

import json

import click

from . import render, speakers
from .config import Config
from .model import Transcript
from .session import Session

TRANSCRIPT_JSON = "transcript.json"


def load_transcript(session: Session) -> Transcript:
    return Transcript.from_dict(json.loads(session.path(TRANSCRIPT_JSON).read_text()))


def process(
    session: Session,
    cfg: Config,
    *,
    fast: bool = False,
    retranscribe: bool = True,
    diarize: bool = True,
    summarize: bool = True,
    visual: bool = True,
) -> None:
    if retranscribe or not session.path(TRANSCRIPT_JSON).exists():
        from .transcribe import transcribe_session

        transcript = transcribe_session(session, cfg, fast=fast)
        if diarize and cfg.diarization.enabled:
            from .diarize import diarize_session

            diarize_session(session, transcript, cfg)
    else:
        transcript = load_transcript(session)

    speakers.identify(transcript, speakers.VoiceDB.load(), cfg)
    render.write_transcript(session, transcript)
    click.echo(f"[transcript] {session.path('transcript.md')}")

    if not summarize:
        return
    keyframes = []
    if visual and session.meta.video:
        from .visual import extract_keyframes

        keyframes = extract_keyframes(session, cfg)
    from .summarize import summarize_session

    summarize_session(session, transcript, cfg, keyframes=keyframes)
