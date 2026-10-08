"""Command-line interface."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import click

from . import pipeline, render, speakers
from .config import load_config
from .session import Session


@click.group()
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Config file (default: ~/.config/meetingrec/config.toml).")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """Record, transcribe, and summarize meetings on Linux."""
    ctx.obj = load_config(config_path)


def _record(cfg, title: str, out_root: Path | None, screen: str | None, mic: str | None, system: str | None) -> Session:
    from .capture.recorder import record

    session = Session.create(out_root or cfg.out_root, title, dt.datetime.now().astimezone())
    record(session, cfg, screen=screen, mic=mic, system=system)
    click.echo(f"\nSaved: {session.dir}")
    return session


_screen_option = click.option(
    "--screen", type=click.Choice(["window", "monitor"]), default=None,
    help="Also record the screen in sync: pick a window (asks every time) or a monitor (remembered).",
)


@cli.command()
@click.argument("title", default="meeting")
@click.option("--out-root", type=click.Path(path_type=Path))
@_screen_option
@click.option("--mic", help="Microphone source (default: system default).")
@click.option("--system", help="System-audio monitor source (default: monitor of default sink).")
@click.pass_obj
def record(cfg, title, out_root, screen, mic, system) -> None:
    """Record only. Ctrl+C stops."""
    _record(cfg, title, out_root, screen, mic, system)


@cli.command()
@click.argument("title", default="meeting")
@click.option("--out-root", type=click.Path(path_type=Path))
@_screen_option
@click.option("--mic")
@click.option("--system")
@click.option("--fast", is_flag=True, help="Use the faster, less accurate ASR model.")
@click.pass_obj
def full(cfg, title, out_root, screen, mic, system, fast) -> None:
    """Record, then transcribe, diarize, and summarize."""
    session = _record(cfg, title, out_root, screen, mic, system)
    pipeline.process(session, cfg, fast=fast)
    click.echo(f"\nDone: {session.dir}")


@cli.command()
@click.argument("meetings", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--fast", is_flag=True, help="Use the faster, less accurate ASR model.")
@click.option("--summary-only", is_flag=True, help="Reuse transcript.json; only re-run naming, rendering, and the summary.")
@click.option("--no-summary", is_flag=True)
@click.option("--no-diarize", is_flag=True)
@click.option("--no-visual", is_flag=True, help="Do not send screen keyframes to the summarizer.")
@click.pass_obj
def process(cfg, meetings, fast, summary_only, no_summary, no_diarize, no_visual) -> None:
    """(Re)process recorded meeting directories, including 0.1 directories with audio.wav."""
    failed = 0
    for m in meetings:
        session = Session.load(m)
        click.echo(f"== {session.dir.name}")
        try:
            pipeline.process(
                session, cfg, fast=fast, retranscribe=not summary_only,
                diarize=not no_diarize, summarize=not no_summary, visual=not no_visual,
            )
        except Exception as exc:  # keep going through a batch
            failed += 1
            click.echo(f"[error] {session.dir.name}: {exc}", err=True)
            if len(meetings) == 1:
                raise
    sys.exit(1 if failed else 0)


@cli.group(name="speakers")
def speakers_group() -> None:
    """Name speakers and manage stored voiceprints."""


@speakers_group.command(name="list")
def speakers_list() -> None:
    """List stored voiceprints."""
    db = speakers.VoiceDB.load()
    for name, n in db.summary():
        click.echo(f"{name}\t{n} sample(s)")


@speakers_group.command(name="name")
@click.argument("meeting", type=click.Path(exists=True, path_type=Path))
@click.argument("label")
@click.argument("name")
@click.option("--no-enroll", is_flag=True, help="Rename in this meeting only; do not store a voiceprint.")
@click.pass_obj
def speakers_name(cfg, meeting, label, name, no_enroll) -> None:
    """Name speaker LABEL (e.g. SPEAKER_01) in MEETING and remember the voice."""
    session = Session.load(meeting)
    transcript = pipeline.load_transcript(session)
    db = speakers.VoiceDB.load()
    speakers.assign_name(transcript, label, name, db=None if no_enroll else db)
    render.write_transcript(session, transcript)
    click.echo(f"{label} -> {name}. Re-run `meetingrec process --summary-only {session.dir}` to refresh the summary.")


@speakers_group.command(name="forget")
@click.argument("name")
def speakers_forget(name) -> None:
    """Delete a stored voiceprint."""
    db = speakers.VoiceDB.load()
    if not db.forget(name):
        raise click.ClickException(f"no voiceprint named {name!r}")


@cli.command()
@click.pass_obj
def doctor(cfg) -> None:
    """Check audio sources, GStreamer, GPU, models, and the Claude CLI."""
    from .doctor import run_doctor

    sys.exit(run_doctor(cfg))
