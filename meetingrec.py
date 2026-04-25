"""meetingrec — record + transcribe + summarize meetings on Linux/PipeWire.

Stack: ffmpeg (mic + system audio) -> faster-whisper (German fine-tune)
       -> claude CLI (Meeting Scribe summary).
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parent
GERMAN_MODEL = PROJECT_ROOT / "models" / "whisper-de-turbo-ct2"
MULTILINGUAL_MODEL_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
DEFAULT_OUT_ROOT = Path.home() / "Recordings" / "meetings"
COMPUTE_TYPE = "int8"  # GTX 1060 (Pascal) — no fp16 ALUs


# ---------- helpers ---------------------------------------------------------


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", text) or "meeting"


def _meeting_dir(slug: str, root: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    p = root / f"{ts}_{_slugify(slug)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_monitor_source() -> str:
    """Resolve the monitor source name for the current default sink."""
    sink = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
    return f"{sink}.monitor"


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# ---------- record ----------------------------------------------------------


def record_audio(out_wav: Path, *, mic: str | None, system: str | None) -> None:
    """Capture mic + system audio, mixed mono 16kHz wav. Stops on Ctrl+C."""
    mic = mic or "@DEFAULT_SOURCE@"
    system = system or _default_monitor_source()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "pulse", "-i", mic,
        "-f", "pulse", "-i", system,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0[a]",
        "-map", "[a]",
        "-ar", "16000", "-ac", "1",
        "-y", str(out_wav),
    ]
    click.echo(f"[record] mic={mic}")
    click.echo(f"[record] sys={system}")
    click.echo(f"[record] -> {out_wav}")
    click.echo("[record] press Ctrl+C to stop\n")

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)  # ffmpeg writes trailer on SIGINT
        proc.wait(timeout=10)
        click.echo("\n[record] stopped")


# ---------- transcribe ------------------------------------------------------


def _load_model(prefer_german: bool):
    from faster_whisper import WhisperModel

    if prefer_german and GERMAN_MODEL.exists():
        click.echo(f"[transcribe] model: {GERMAN_MODEL.name} (de fine-tune)")
        return WhisperModel(str(GERMAN_MODEL), device="cuda", compute_type=COMPUTE_TYPE)

    click.echo(f"[transcribe] model: {MULTILINGUAL_MODEL_ID} (multilingual)")
    return WhisperModel(MULTILINGUAL_MODEL_ID, device="cuda", compute_type=COMPUTE_TYPE)


def transcribe_audio(audio: Path, out_dir: Path, *, language: str | None, multilingual: bool) -> tuple[Path, Path, str]:
    """Run faster-whisper. Returns (srt_path, txt_path, detected_lang)."""
    prefer_german = not multilingual and (language is None or language == "de")
    model = _load_model(prefer_german)

    t0 = time.time()
    segments, info = model.transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    detected = info.language
    click.echo(f"[transcribe] detected lang: {detected} (p={info.language_probability:.2f})")

    # If detection says English but we loaded the German fine-tune, reload multilingual.
    if prefer_german and detected != "de":
        click.echo("[transcribe] non-German detected — reloading multilingual model")
        del model
        model = _load_model(prefer_german=False)
        segments, info = model.transcribe(
            str(audio),
            language=detected,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

    srt_path = out_dir / "transcript.srt"
    txt_path = out_dir / "transcript.txt"
    with srt_path.open("w") as srt, txt_path.open("w") as txt:
        for i, seg in enumerate(segments, 1):
            srt.write(f"{i}\n{_format_ts(seg.start)} --> {_format_ts(seg.end)}\n{seg.text.strip()}\n\n")
            txt.write(seg.text.strip() + "\n")

    click.echo(f"[transcribe] done in {time.time() - t0:.1f}s -> {txt_path.name}")
    return srt_path, txt_path, detected


# ---------- summarize -------------------------------------------------------


SCRIBE_PROMPT_DE = """Du bist ein erfahrener Meeting-Protokollant. Erstelle ein strukturiertes Protokoll \
aus dem folgenden Transkript. Antworte ausschließlich auf Deutsch.

Format (Markdown):

# Zusammenfassung
2–4 Sätze: Worum ging es, was war das Ergebnis.

# Wichtigste Punkte
- Stichpunkte zu zentralen Themen, Entscheidungen, Hintergründen.

# Entscheidungen
- Was wurde entschieden? Wenn nichts entschieden wurde, schreibe „keine".

# To-dos
| Aufgabe | Verantwortlich | Frist |
|---|---|---|
Wenn keine klare Frist genannt wurde, trage „—" ein und markiere die Zeile mit ⚠️.
Wenn niemand benannt wurde, trage „?" ein und markiere mit ⚠️.

# Offene Fragen
- Punkte, die im Meeting nicht geklärt wurden.

Transkript:
"""

SCRIBE_PROMPT_EN = """You are an experienced meeting scribe. Produce structured minutes \
from the transcript below. Respond in English.

Format (Markdown):

# Summary
2–4 sentences: what it was about, what the outcome was.

# Key Points
- Bullet points covering central topics, decisions, background.

# Decisions
- What was decided? If nothing, write "none".

# Action Items
| Task | Owner | Due |
|---|---|---|
If no clear due date was named, put "—" and mark the row with ⚠️.
If no owner was named, put "?" and mark with ⚠️.

# Open Questions
- Points left unresolved.

Transcript:
"""


def summarize_transcript(txt_path: Path, out_dir: Path, *, language: str) -> Path:
    prompt = SCRIBE_PROMPT_DE if language == "de" else SCRIBE_PROMPT_EN
    transcript = txt_path.read_text()
    full_input = prompt + "\n" + transcript

    click.echo(f"[summarize] -> claude (~{len(transcript)} chars, lang={language})")
    t0 = time.time()
    result = subprocess.run(
        ["claude", "--print", "--effort", "low"],
        input=full_input, text=True, capture_output=True, check=True,
    )
    summary_path = out_dir / "summary.md"
    summary_path.write_text(result.stdout)
    click.echo(f"[summarize] done in {time.time() - t0:.1f}s -> {summary_path.name}")
    return summary_path


# ---------- CLI -------------------------------------------------------------


@click.group()
def cli() -> None:
    """Record + transcribe + summarize meetings."""


@cli.command()
@click.argument("name", default="meeting")
@click.option("--out-root", type=click.Path(path_type=Path), default=DEFAULT_OUT_ROOT)
@click.option("--mic", help="PulseAudio source for mic. Default: system default mic.")
@click.option("--system", help="PulseAudio source for system audio. Default: default sink monitor.")
def record(name: str, out_root: Path, mic: str | None, system: str | None) -> None:
    """Record only — Ctrl+C to stop."""
    d = _meeting_dir(name, out_root)
    record_audio(d / "audio.wav", mic=mic, system=system)
    click.echo(f"\nSaved: {d}")


@cli.command()
@click.argument("audio", type=click.Path(exists=True, path_type=Path))
@click.option("--language", help="Force language (e.g. de, en). Default: auto-detect.")
@click.option("--multilingual", is_flag=True, help="Skip German fine-tune; use multilingual model.")
def transcribe(audio: Path, language: str | None, multilingual: bool) -> None:
    """Transcribe an existing wav/mp3 file."""
    out = audio.parent
    transcribe_audio(audio, out, language=language, multilingual=multilingual)


@cli.command()
@click.argument("transcript", type=click.Path(exists=True, path_type=Path))
@click.option("--language", default="de", help="Output language (de/en). Default: de.")
def summarize(transcript: Path, language: str) -> None:
    """Summarize an existing transcript.txt via Claude."""
    summarize_transcript(transcript, transcript.parent, language=language)


@cli.command()
@click.argument("name", default="meeting")
@click.option("--out-root", type=click.Path(path_type=Path), default=DEFAULT_OUT_ROOT)
@click.option("--mic", help="PulseAudio source for mic.")
@click.option("--system", help="PulseAudio source for system audio.")
@click.option("--language", help="Force language. Default: auto-detect.")
@click.option("--multilingual", is_flag=True, help="Use multilingual model.")
def full(name: str, out_root: Path, mic: str | None, system: str | None,
         language: str | None, multilingual: bool) -> None:
    """Record -> transcribe -> summarize in one shot."""
    d = _meeting_dir(name, out_root)
    audio = d / "audio.wav"
    record_audio(audio, mic=mic, system=system)
    if not audio.exists() or audio.stat().st_size < 1024:
        click.echo("[full] no audio captured; aborting", err=True)
        sys.exit(1)
    _, txt, detected = transcribe_audio(audio, d, language=language, multilingual=multilingual)
    summarize_transcript(txt, d, language=detected if detected in ("de", "en") else "en")
    click.echo(f"\nDone: {d}")


if __name__ == "__main__":
    cli()
