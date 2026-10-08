# meetingrec

Local meeting recorder for Linux desktops. It records your microphone, the call's
audio and (optionally) the screen in one synchronized file. It then transcribes each
track with Whisper, tells the remote speakers apart, recognizes voices it has met
before, and asks the Claude CLI for structured minutes. The tool checks those minutes
against the transcript before writing them.

German-first, works for any Whisper language. Everything except the summary runs locally.

## What you get per meeting

```text
~/Recordings/meetings/2026-10-08_141500_kickoff/
  meeting.json      start time, tracks, sources
  recording.mka     mic + system audio as two separate FLAC tracks
  recording.mkv     … plus the screen as H.264 when recorded with --screen
  notes.md          optional; your own notes, treated as authoritative by the summary
  transcript.md     speaker-labelled transcript with timestamps
  transcript.{txt,srt,vtt,json}
  summary.md        minutes: summary, key points, decisions, to-dos, open questions
  summary.json      the structured minutes plus validation results
  frames/           screen keyframes the summary looked at
```

## How it works

1. **Capture**: one GStreamer pipeline writes the microphone and the system-audio
   monitor as separate tracks. With `--screen`, it also writes the PipeWire screen stream
   from the xdg-desktop-portal ScreenCast API, encoded with NVENC or x264. All tracks
   share one clock, so audio, video and transcript timestamps line up exactly.
2. **Transcribe**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs
   `large-v3` on each track (`--fast` uses `large-v3-turbo`), with Silero VAD, word
   timestamps, and your glossary as hotwords. The mic track is you. If the mic picks
   up words that were played over loudspeakers at the same moment, they are dropped.
3. **Diarize**: [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   separates the remote speakers on the system track only.
4. **Identify**: each remote speaker's voice embedding is compared with the voiceprints
   you have stored. Known voices get their name. Unknown ones stay `SPEAKER_NN` until
   you name them once.
5. **Summarize**: the Claude CLI returns schema-checked JSON. The prompt includes:
   - the meeting date, weekday and a calendar
   - the list of known participant names
   - your glossary and `notes.md`
   - the screen keyframes

   meetingrec then checks every item mechanically:
   - owners must be known participants
   - due dates must match the weekday that was said
   - timestamps must exist
   - quotes must appear in the transcript near their timestamp

   Anything that fails is marked ⚠️ instead of being trusted.

## Requirements

- Linux with PipeWire (or PulseAudio) and GStreamer 1.22+ with the `good`/`bad`/`ugly`
  plugin sets and `gstreamer1.0-pipewire` (screen capture). On Debian/Ubuntu:
  `sudo apt install gstreamer1.0-pipewire gstreamer1.0-plugins-{good,bad,ugly} ffmpeg pulseaudio-utils`
- A desktop with the xdg-desktop-portal ScreenCast API (GNOME, KDE) for `--screen`.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU is strongly recommended. `large-v3` needs about 4 GB of VRAM in int8 and
  runs at about 0.1× real time on a GTX 1060. CPU works, but slowly.
- The [Claude CLI](https://docs.claude.com/en/docs/claude-code), logged in, for summaries.

## Install

```bash
git clone https://github.com/TueJon/meetingrec.git
cd meetingrec
uv sync                    # transcription only
uv sync --extra diarize    # + speaker diarization (PyTorch, ~3 GB)
ln -s "$PWD/meetingrec" ~/.local/bin/meetingrec
meetingrec doctor
```

Diarization uses a gated model. Accept its conditions on
<https://huggingface.co/pyannote/speaker-diarization-community-1>, then run
`uvx hf auth login` once. Without it, the remote speakers show as `Remote`.

The `diarize` extra installs PyTorch from the CUDA 12.6 wheel index. That is the last
one that still supports Pascal GPUs (GTX 10xx).

## Usage

```bash
meetingrec full kickoff                  # record (Ctrl+C stops), then process
meetingrec full demo --screen window     # also record one window (picker every time)
meetingrec full demo --screen monitor    # record a monitor (picked once, remembered)
meetingrec record customer-call          # record now …
meetingrec process ~/Recordings/meetings/2026-10-08_*   # … process later (many dirs ok)
meetingrec process --summary-only <dir>  # re-run naming, rendering and the summary
meetingrec process --fast <dir>          # large-v3-turbo instead of large-v3
```

Name a speaker once and they are recognized in later meetings:

```bash
meetingrec speakers name <dir> SPEAKER_01 "Thomas"
meetingrec process --summary-only <dir>
meetingrec speakers list
meetingrec speakers forget "Thomas"
```

Directories from meetingrec 0.1 (a single mixed `audio.wav`) can be processed too. They
have no separate mic track, so every speaker goes through diarization.

## Configuration

Optional: `~/.config/meetingrec/config.toml`. All keys are optional; the values below
are the defaults unless noted.

```toml
self_name = "Jonas"               # your name in transcripts (default "Ich"/"Me")
language = "auto"                 # or "de", "en", …
timezone = ""                     # IANA name for the summary's calendar; "" = local
glossary = ["Narev", "Keycloak"]  # names + jargon, spelled right (hotwords + summary)
participants = ["Thomas"]         # people who may own action items

[capture]
mic = ""                          # source name; "" = default source
system = ""                       # monitor source; "" = monitor of the default sink
video_fps = 2

[asr]
model = "large-v3"                # alias or any faster-whisper model id / path
fast_model = "large-v3-turbo"
device = "auto"                   # auto | cuda | cpu

[diarization]
enabled = true
max_speakers = 6                  # optional hints (default: not set)

[speakers]
match_threshold = 0.55            # cosine similarity needed to auto-name a voice
match_margin = 0.08               # … and lead over the second-best voiceprint

[summary]
model = ""                        # Claude CLI model alias; "" = CLI default
effort = "medium"
language = "auto"                 # auto | de | en
max_keyframes = 12
```

List audio sources with `pactl list short sources`.

## Privacy

This tool records conversations. Tell participants and follow your local law.

- Recordings, transcripts and summaries stay under `~/Recordings/meetings/`. The
  summary step sends the transcript, your notes and the selected keyframes to Claude
  through your own Claude CLI login.
- `--screen window` asks for a window every meeting. `--screen monitor` remembers the
  monitor and records everything that appears on it, including notifications.
- Voiceprints are biometric data. They are stored in
  `~/.local/share/meetingrec/voices.json` (mode 600) and only ever compared locally.

## Troubleshooting

Run `meetingrec doctor` first. It checks GStreamer elements, audio sources, the
ScreenCast portal, the GPU, cached models, Hugging Face access and the Claude CLI.

- **Transcript misses speech or garbles names**: add the names and jargon to `glossary`.
  Prefer `large-v3` over turbo. Avoid Whisper fine-tunes that do not keep timestamp
  tokens: they can silently skip whole 30 s windows of long recordings.
- **Your words appear twice**: use a headset. Echo filtering only removes near-identical
  text.
- **Screen picker appears every time with `--screen monitor`**: the stored token is
  single-use and is renewed on every run. If the monitor layout changed, pick again once.

## License

MIT. See [LICENSE](LICENSE). Model weights come with their own licenses: Whisper is
MIT, and pyannote community-1 is CC-BY-4.0, gated.
