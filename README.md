# meetingrec

Local meeting recorder + transcriber + Claude summarizer. German-first, runs on the GTX 1060.

## Where things live

| What | Path |
|---|---|
| Project | `~/tools/meetingrec/` |
| CLI binary (symlink) | `~/.local/bin/meetingrec` |
| Wrapper script | `~/tools/meetingrec/meetingrec` |
| Main code | `~/tools/meetingrec/meetingrec.py` |
| German Whisper model (CT2 fp16) | `~/tools/meetingrec/models/whisper-de-turbo-ct2/` |
| HF cache (extra models) | `~/.cache/huggingface/` |
| Recordings + transcripts + summaries | `~/Recordings/meetings/<timestamp>_<slug>/` |

## Usage

```bash
# Record + transcribe + summarize, Ctrl+C to stop recording
meetingrec full kickoff-tobias

# Just record (e.g. on the road), transcribe later
meetingrec record drive-back
meetingrec transcribe ~/Recordings/meetings/2026-04-25_*/audio.wav
meetingrec summarize ~/Recordings/meetings/2026-04-25_*/transcript.txt

# Force English (skip German fine-tune)
meetingrec full standup --multilingual
meetingrec transcribe audio.wav --language en --multilingual

# Custom output dir
meetingrec full demo --out-root /tmp/demo

# See all commands
meetingrec --help
meetingrec full --help
```

Each meeting dir contains:

```
audio.wav         16kHz mono mix of mic + system audio
transcript.srt    timestamped subtitles
transcript.txt    plain text
summary.md        Claude-generated, structured (Zusammenfassung / Entscheidungen / To-dos / Offene Fragen)
```

## How it works

1. **Capture** — `ffmpeg` with two PulseAudio inputs (default mic + default sink monitor), mixed via `amix`, 16kHz mono.
2. **Transcribe** — `faster-whisper` loads `primeline/whisper-large-v3-turbo-german` (converted to CTranslate2 fp16) on CUDA, int8 compute (Pascal limitation). Runs language detection on the first 30s of audio; if it's not German, reloads the multilingual `mobiuslabsgmbh/faster-whisper-large-v3-turbo` and re-transcribes.
3. **Summarize** — pipes the transcript to `claude --print --effort low` with a German Meeting-Scribe prompt (or English equivalent). Uses your Claude Code OAuth, not an API key.

## Audio sources

Capture uses your **system default mic** and **default sink monitor** by default. Override:

```bash
# List sources
pactl list short sources

# Use a specific mic + system audio combo
meetingrec record meeting --mic alsa_input.usb-Auna_Mic_CM900_...mono-fallback \
                          --system bluez_output.04_00_6E_CC_19_A5.1.monitor
```

If you switch your default audio device mid-meeting, audio capture won't follow — it locks the source on start.

## Known gotchas

- **Pascal GPU (GTX 1060)** can't do fp16 ALU efficiently — `compute_type` is hard-coded to `int8`. On an RTX 2060+ you'd switch to `float16` for 3-5× speed.
- **No diarization** ("who said what"). Adding pyannote 3.1 needs a HuggingFace token + accepting model terms on the website. Not wired up.
- **Claude billing** runs through Claude Code OAuth, not the API. Each summary call costs whatever a small Sonnet print call costs on your plan.
- **`--bare` doesn't work** for the summary subprocess — Claude Code in `--bare` mode requires `ANTHROPIC_API_KEY`, which isn't set. We use plain `--print` so OAuth applies.

## Reinstall / reset

```bash
cd ~/tools/meetingrec
uv sync                                  # rebuild venv from pyproject.toml
ls models/whisper-de-turbo-ct2/          # German model lives here, ignored from git
```

If the German model is missing, regenerate:

```bash
cd ~/tools/meetingrec
uv run ct2-transformers-converter \
  --model primeline/whisper-large-v3-turbo-german \
  --output_dir models/whisper-de-turbo-ct2 \
  --copy_files preprocessor_config.json generation_config.json tokenizer_config.json \
               vocab.json merges.txt normalizer.json added_tokens.json special_tokens_map.json \
  --quantization float16 --force
```
