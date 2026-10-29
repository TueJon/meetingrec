# Agent Instructions

`meetingrec` is a Python CLI for local meeting capture, Whisper transcription, diarization, voiceprint speaker naming and validated Claude CLI minutes. Module notes (timeline convention, track tagging): `docs/architecture.md`. Checks: `uv run pytest -q`, `uv run ruff check`. Smoke tests with `--screen` need a human at the desktop (portal picker, or an xdotool click on X11).

## Conventions

- Measure ASR changes on real audio, never by reading the text. Compare words per
  speech-minute and coverage against a baseline. A fluent transcript can be missing
  half the speech (this happened with a German Whisper fine-tune in 0.1).
  Produce the audio with `meetingrec record smoke --out-root /tmp/mr`, then
  `meetingrec process /tmp/mr/*` (needs real audio, a GPU and the Claude CLI).
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
