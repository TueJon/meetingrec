"""Command-line interface."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import click

from . import pipeline, render, speakers
from .config import load_config
from .session import Session


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    help="Config file (default: ~/.config/meetingrec/config.toml).",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """Record, transcribe, and summarize meetings on Linux."""
    ctx.obj = load_config(config_path)


def _record(cfg, title: str, out_root: Path | None, events: bool, **options) -> Session:
    from .capture.portal import ScreenCastCancelled
    from .capture.recorder import json_event, print_event, record

    emit = json_event if events else print_event
    session = Session.create(out_root or cfg.out_root, title, dt.datetime.now().astimezone())
    try:
        record(session, cfg, emit=emit, **options)
    except ScreenCastCancelled:
        raise click.ClickException("screen selection cancelled; nothing was recorded") from None
    except RuntimeError:
        if not any(session.dir.iterdir()):
            session.dir.rmdir()
        raise
    emit({"event": "saved", "dir": str(session.dir)})
    click.echo(f"\nSaved: {session.dir}")
    return session


def _capture_options(fn):
    for option in reversed(
        [
            click.argument("title", default="meeting"),
            click.option("--out-root", type=click.Path(path_type=Path)),
            click.option(
                "--screen",
                type=click.Choice(["window", "monitor"]),
                default=None,
                help="Also record the screen in sync: pick a window (asks every time) "
                "or a monitor (remembered).",
            ),
            click.option(
                "--window-id", type=int, help="X11: record this window instead of asking for a click."
            ),
            click.option("--monitor", help="X11: record this monitor (xrandr name) instead of asking."),
            click.option("--mic", help="Microphone source (default: system default)."),
            click.option("--system", help="System-audio monitor source (default: monitor of default sink)."),
            click.option(
                "--app",
                help="Record only this application's audio instead of the system mix: a PID (e.g. of the "
                "call window) or a binary name (chrome, teams-for-linux, Discord). See `meetingrec targets`.",
            ),
            click.option(
                "--room",
                is_flag=True,
                help="In-person meeting: record the microphone only and diarize everyone on it.",
            ),
            click.option(
                "--events", is_flag=True, help="Print JSON-lines events (levels, warnings) on stdout."
            ),
        ]
    ):
        fn = option(fn)
    return fn


@cli.command()
@_capture_options
@click.pass_obj
def record(cfg, title, out_root, events, **options) -> None:
    """Record only. Ctrl+C stops."""
    _record(cfg, title, out_root, events, **options)


@cli.command()
@_capture_options
@click.option("--fast", is_flag=True, help="Use the faster, less accurate ASR model.")
@click.pass_obj
def full(cfg, title, out_root, events, fast, **options) -> None:
    """Record, then transcribe, diarize, and summarize."""
    session = _record(cfg, title, out_root, events, **options)
    if events:
        click.echo(json.dumps({"event": "processing", "dir": str(session.dir)}))
    pipeline.process(session, cfg, fast=fast)
    if events:
        click.echo(json.dumps({"event": "done", "dir": str(session.dir)}))
    click.echo(f"\nDone: {session.dir}")


@cli.command()
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def targets(cfg, as_json) -> None:
    """List what can be recorded: microphones, apps playing audio, windows and monitors."""
    from .capture.targets import collect

    found = {**collect(), "out_root": str(cfg.out_root)}
    if as_json:
        click.echo(json.dumps(found, indent=2))
        return
    click.echo(f"session: {found['session']}")
    click.echo("microphones:")
    for m in found["mics"]:
        click.echo(f"  {'*' if m['default'] else ' '} {m['name']}  ({m['description']})")
    click.echo("apps playing audio (--app):")
    for a in found["apps"]:
        click.echo(f"    {a['binary']:<20} pid {a['pid']}  {a['app']}: {a['media']}")
    click.echo("windows (--window-id, --app <pid>):")
    for w in found["windows"]:
        click.echo(f"    {w['id']:<10} pid {w['pid']:<8} {'♪ ' if w['audio'] else '  '}{w['title']}")
    click.echo("monitors (--monitor):")
    for m in found["monitors"]:
        click.echo(f"    {m['name']}  {m['width']}x{m['height']}")


@cli.command()
def gui() -> None:
    """Open the recorder window (quick start or a guided setup)."""
    from .gui_launcher import launch

    launch()


@cli.command()
@click.argument("meetings", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--fast", is_flag=True, help="Use the faster, less accurate ASR model.")
@click.option(
    "--summary-only",
    is_flag=True,
    help="Reuse transcript.json; only re-run naming, rendering, and the summary.",
)
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
                session,
                cfg,
                fast=fast,
                retranscribe=not summary_only,
                diarize=not no_diarize,
                summarize=not no_summary,
                visual=not no_visual,
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
    click.echo(
        f"{label} -> {name}. Re-run `meetingrec process --summary-only {session.dir}` to refresh the summary."
    )


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
