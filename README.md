# meetingrec

Local meeting recorder, Whisper transcription, and Claude-powered meeting-summary CLI
for Linux desktops.

`meetingrec` records microphone audio plus system audio, writes a local WAV file,
transcribes it with `faster-whisper`, and asks the Claude CLI to produce structured
meeting minutes. It is German-first by default and falls back to a multilingual
Whisper model when the detected language is not German.

## Features

- Records microphone and system audio together via `ffmpeg` and PulseAudio/PipeWire.
- Produces `audio.wav`, `transcript.srt`, `transcript.txt`, and `summary.md`.
- Uses a local CTranslate2 German Whisper model when available.
- Falls back to a multilingual faster-whisper model for non-German meetings.
- Summarizes through the installed Claude CLI; no API key is stored in this repo.

## Requirements

- Linux desktop with PipeWire or PulseAudio-compatible sources.
- Python 3.12.
- `uv` for dependency management.
- `ffmpeg` and `pactl`.
- NVIDIA GPU with CUDA support for the default `device="cuda"` configuration.
- Claude CLI installed and authenticated, or an `ANTHROPIC_API_KEY` available to the
  Claude CLI if you use API-key auth.

Install system tools on Debian/Ubuntu-style systems:

```bash
sudo apt install ffmpeg pulseaudio-utils
```

Install `uv` from the official Astral instructions if it is not already available.

## Install

```bash
git clone https://github.com/tue-Jonas/meetingrec.git
cd meetingrec
uv sync
```

Run the CLI directly:

```bash
uv run meetingrec --help
```

Optional shell shortcut:

```bash
ln -s "$PWD/meetingrec" ~/.local/bin/meetingrec
meetingrec --help
```

## Configuration

The tool does not read a `.env` file automatically. Use your shell environment for
optional values consumed by the external tools:

```bash
cp .env.example .env
# Edit .env if needed, then:
set -a
. ./.env
set +a
```

Useful environment variables:

- `ANTHROPIC_API_KEY`: optional, only needed if your Claude CLI uses API-key auth
  instead of its normal logged-in OAuth session.
- `HF_TOKEN`: optional, only needed for private or gated Hugging Face models.

Generated recordings, transcripts, summaries, local env files, virtualenvs, and model
weights are ignored by git.

## Usage

Record, transcribe, and summarize in one command. Press `Ctrl+C` to stop recording:

```bash
uv run meetingrec full kickoff
```

Record now and process later:

```bash
uv run meetingrec record customer-call
uv run meetingrec transcribe ~/Recordings/meetings/2026-04-25_*/audio.wav
uv run meetingrec summarize ~/Recordings/meetings/2026-04-25_*/transcript.txt
```

Force English output or skip the German model:

```bash
uv run meetingrec full standup --multilingual
uv run meetingrec transcribe audio.wav --language en --multilingual
uv run meetingrec summarize transcript.txt --language en
```

Choose a different output root:

```bash
uv run meetingrec full demo --out-root /tmp/meetingrec-demo
```

Each meeting directory contains:

```text
audio.wav       16 kHz mono mix of mic + system audio
transcript.srt  timestamped subtitles
transcript.txt  plain transcript
summary.md      structured Claude-generated minutes
```

## Audio Sources

By default, capture uses the system default microphone and the monitor source for the
current default sink. List sources with:

```bash
pactl list short sources
```

Use explicit sources when needed:

```bash
uv run meetingrec record meeting \
  --mic alsa_input.usb-Example_Microphone.mono-fallback \
  --system bluez_output.00_11_22_33_44_55.1.monitor
```

If the default audio device changes during a meeting, the running `ffmpeg` process keeps
using the sources selected at startup.

## German Whisper Model

`meetingrec` looks for a local German CTranslate2 model at:

```text
models/whisper-de-turbo-ct2/
```

If that directory is absent, it uses the multilingual faster-whisper model ID configured
in `meetingrec.py`.

To create the German model locally:

```bash
uv run ct2-transformers-converter \
  --model primeline/whisper-large-v3-turbo-german \
  --output_dir models/whisper-de-turbo-ct2 \
  --copy_files preprocessor_config.json generation_config.json tokenizer_config.json \
               vocab.json merges.txt normalizer.json added_tokens.json special_tokens_map.json \
  --quantization float16 --force
```

Model files are large and are intentionally ignored by git.

## Troubleshooting

- `ffmpeg` cannot open an input: run `pactl list short sources` and pass explicit
  `--mic` and `--system` values.
- No system audio: use a `.monitor` source for `--system`.
- CUDA errors: verify the NVIDIA driver/CUDA runtime and consider changing
  `device="cuda"` or `COMPUTE_TYPE` in `meetingrec.py` for your hardware.
- Claude summary fails: run `claude --print "hello"` to verify the Claude CLI is
  installed and authenticated in the same shell.
- Diarization is not implemented. Adding speaker labels would require a separate
  diarization model and its own license/token review.

## License

MIT. See [LICENSE](LICENSE).
