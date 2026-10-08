# Agent Instructions

`meetingrec` is a Python CLI for local meeting capture (mic, system audio, screen),
Whisper transcription, pyannote diarization, voiceprint-based speaker naming, and
validated Claude CLI minutes.

## Architecture

`src/meetingrec/`:

- `model.py`: the shared data model (`Transcript`, `Segment`, `Word`, `SpeakerInfo`,
  `Keyframe`). All times are seconds on the meeting timeline, where 0 is the moment
  the capture pipeline started.
- `config.py`: `~/.config/meetingrec/config.toml`. Unknown keys are errors.
- `session.py`: the meeting directory and `meeting.json`. Also extracts each track to
  a 16 kHz WAV, with its start offset, and loads 0.1 directories (`audio.wav`) as a
  single `mixed` track.
- `capture/portal.py`: the xdg-desktop-portal ScreenCast handshake (jeepney).
- `capture/recorder.py`: one `gst-launch-1.0 -e` pipeline writes `recording.mka` or
  `recording.mkv`. Audio tracks are tagged `mic`/`system`, and the stream indices are
  mapped from those tags after the recording stops.
- `transcribe.py`: faster-whisper per track, plus echo dropping on the mic track.
- `diarize.py`: pyannote on the `system` (or `mixed`) track, and assignment of
  speakers per word.
- `speakers.py`: the voiceprint DB (`~/.local/share/meetingrec/voices.json`) and
  naming.
- `render.py`: the transcript outputs and `transcript_lines()`, which the summary
  consumes.
- `visual.py`: screen keyframes for the summary.
- `summarize.py`: the Claude CLI call (`--json-schema`), deterministic validation,
  and the summary renderers.
- `pipeline.py`: processing order. `cli.py`: Click commands. `doctor.py`:
  environment checks.

## Commands

```bash
uv sync --group dev            # add --extra diarize for pyannote/PyTorch
uv run pytest -q
uv run ruff check src tests && uv run ruff format --check src tests
uv run meetingrec doctor
```

Manual smoke tests need real audio, a GPU or the Claude CLI:
`meetingrec record smoke --out-root /tmp/mr`, then `meetingrec process /tmp/mr/*`.
`--screen` opens the desktop's screen picker, so it can only be tested with a human
at the machine.

## Conventions

- Measure ASR changes on real audio, never by reading the text. Compare words per
  speech-minute and coverage against a baseline. A fluent transcript can be missing
  half the speech (this happened with a German Whisper fine-tune in 0.1).
- Summary facts must be checkable. Anything the model could get wrong mechanically
  (owners, dates, timestamps, quotes) is validated in Python and flagged, never
  silently trusted.
- Keep the CLI dependency-light. Add no services, daemons or databases. Heavy
  optional features go behind extras.
- Never hardcode user home directories, hostnames, customer names, or private paths.
- Keep German output strong by default and preserve the English path.
- When dependencies change, run `uv lock`. Check licenses: no avoidable copyleft or
  proprietary runtime dependency.

## Security Notes

- This tool records private conversations and stores voiceprints, which are
  biometric data. Treat test recordings, transcripts and `voices.json` as sensitive
  local data.
- The repo must never contain API keys or tokens, `.env` files, recordings,
  transcripts, summaries, frames, or voiceprints.
- Before a release, scan the working tree and the git history for secrets and
  private meeting artifacts.
