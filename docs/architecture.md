# Architecture

`src/meetingrec/`:

- `model.py`: the shared data model (`Transcript`, `Segment`, `Word`, `SpeakerInfo`,
  `Keyframe`). All times are seconds on the meeting timeline, where 0 is the moment
  the capture pipeline started.
- `config.py`: `~/.config/meetingrec/config.toml`. Unknown keys are errors.
- `session.py`: the meeting directory and `meeting.json`. Also extracts each track to
  a 16 kHz WAV, with its start offset, and loads 0.1 directories (`audio.wav`) as a
  single `mixed` track.
- `capture/portal.py`: the xdg-desktop-portal ScreenCast handshake (jeepney).
- `capture/x11.py`: window/monitor selection with xdotool/xrandr on X11 sessions, recorded
  with `ximagesrc` (the portal shows no picker on GNOME/X11).
- `capture/apps.py`: per-application audio. A null sink is the capture bus; a watcher
  thread links the chosen process tree's playback streams into it (`pw-link`).
- `capture/targets.py`: mics, playing apps, windows and monitors for `targets --json`.
- `capture/recorder.py`: one `gst-launch-1.0 -e` pipeline writes `recording.mka` or
  `recording.mkv`. Audio tracks are tagged `mic`/`system`, and the stream indices are
  mapped from those tags after the recording stops. Roles: `mic` + `system`, or `room`
  alone. `audiorate` keeps dropouts as silence; `level` messages (`gst-launch -m`) feed
  live meters and the dropout check; `--events` prints them as JSON lines.
- `gui.py`: GTK 4 / libadwaita window, run by the system Python (`gui_launcher.py`); it
  imports nothing from the package and drives the CLI (`targets --json`, `--events`).
- `transcribe.py`: faster-whisper per track (no VAD on a `room` track), plus echo dropping on the mic track.
- `diarize.py`: pyannote on the `system` (or `mixed`, or `room`) track, and assignment of
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
