# Agent Instructions

`meetingrec` is a small Python CLI for local meeting recording, Whisper
transcription, and Claude CLI summarization.

## Architecture

- `meetingrec.py` contains the Click command group and all runtime behavior.
- `meetingrec` is a portable shell wrapper for running the CLI through `uv`.
- `pyproject.toml` defines Python 3.12 dependencies and the `meetingrec` console
  script.
- `uv.lock` is committed and should be updated whenever dependencies change.
- `models/whisper-de-turbo-ct2/` is an optional local CTranslate2 model directory and
  must never be committed.

Runtime flow:

1. `record_audio()` captures microphone and system audio with `ffmpeg` PulseAudio
   inputs and writes a 16 kHz mono WAV file.
2. `transcribe_audio()` loads faster-whisper/CTranslate2, writes `transcript.srt` and
   `transcript.txt`, and reloads a multilingual model when German-first detection does
   not fit.
3. `summarize_transcript()` sends the transcript to `claude --print --effort low` and
   writes `summary.md`.

## Commands

```bash
uv sync
uv run meetingrec --help
uv run python -m compileall meetingrec.py
```

Manual smoke commands that require local audio/GPU/Claude setup:

```bash
uv run meetingrec record smoke --out-root /tmp/meetingrec-smoke
uv run meetingrec transcribe /path/to/audio.wav --multilingual
uv run meetingrec summarize /path/to/transcript.txt --language de
```

## Conventions

- Keep the CLI dependency-light. Do not add services, daemons, databases, or network
  APIs without a clear product reason.
- Prefer explicit command-line options over hidden machine-specific defaults.
- Do not hardcode user home directories, hostnames, customer names, or private TWB
  paths.
- Keep generated meeting artifacts out of git: audio, transcripts, summaries, local
  model weights, Hugging Face caches, and `.env` files are ignored for a reason.
- If a change touches dependencies, run `uv lock` and check that no avoidable
  proprietary or copyleft runtime dependency was introduced.
- If changing prompts, keep German output strong by default and preserve the English
  path for non-German transcripts.

## Security Notes

- This tool records private conversations. Treat test recordings and transcripts as
  sensitive local data.
- The repo must remain free of API keys, Claude/OpenAI tokens, Hugging Face tokens,
  `.env` files, recordings, transcripts, and summaries.
- Before making the repository public or cutting a release, scan both the working tree
  and git history for secrets and private meeting artifacts.
