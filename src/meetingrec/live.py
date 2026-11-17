"""Live mode: transcribe the call while it runs and let a quick model research hints.

The recorder tees every audio track into `live/<role>.pcm` (16 kHz mono s16le) next to
the recording. `Transcriber` reads what is new every `live.interval` seconds and appends
lines to `live/transcript.txt`. `Hinter` sends the recent transcript every
`live.hint_interval` seconds to the Claude CLI, which runs read-only in `live.workdir`
(the user's knowledge base) and may answer with a few hints. Hints are printed, sent as
`hint` events and appended to `live/hints.md`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

LIVE_DIR = "live"
RATE = 16000
BYTES_PER_SECOND = RATE * 2
MIN_CHUNK_SECONDS = 2.0
# A segment ending this close to the end of the audio read so far is probably cut mid-word;
# it is transcribed again with the next chunk.
CUT_MARGIN_SECONDS = 1.0
HINT_TIMEOUT_S = 180
KINDS = ["answer", "ask", "mention", "contradiction", "remind"]

Emit = Callable[[dict], None]

HINT_SCHEMA = {
    "type": "object",
    "properties": {
        "hints": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": KINDS},
                    "text": {"type": "string"},
                    "source": {"type": "string", "description": "file path or reference backing the hint"},
                },
                "required": ["kind", "text", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["hints"],
    "additionalProperties": False,
}

PROMPT = """You are {name}'s silent assistant during a meeting that is running right now.
Below is the live transcript of the last minutes: "{me}" is {name}, "Remote" is everyone
else on the call. The transcript comes from fast speech recognition and has errors.
Your working directory is {name}'s knowledge base (notes, open threads, history). Use your
tools to look things up when the conversation touches something that may be recorded there.

Give {name} a hint only when it helps right now:
- answer: a question was asked that the records answer
- mention: a fact {name} should bring up
- ask: something {name} should ask or clarify before the topic closes
- contradiction: something said contradicts the records
- remind: an open item with these people that has not come up yet

No hint is the normal answer. Never repeat or rephrase an earlier hint. Each hint at most
25 words, in the language the meeting is held in (German meeting: German hint), concrete
(names, numbers, dates), with the file path or reference it is based on as the source.

Meeting: {title}

Earlier hints:
{earlier}

Transcript (lines after ">>> new" are new since your last look):
{transcript}
"""


def live_dir(session_dir: Path) -> Path:
    return session_dir / LIVE_DIR


@dataclass
class Line:
    start: float
    speaker: str
    text: str

    def render(self) -> str:
        m, s = divmod(int(self.start), 60)
        return f"[{m:02d}:{s:02d}] {self.speaker}: {self.text}"


@dataclass
class Transcriber:
    """Turns the growing PCM files into transcript lines, one chunk per call."""

    model: object
    directory: Path
    roles: list[str]
    language: str
    speakers: dict[str, str]
    offsets: dict[str, int] = field(default_factory=dict)

    def tick(self) -> list[Line]:
        import numpy as np

        lines = []
        for role in self.roles:
            path = self.directory / f"{role}.pcm"
            if not path.exists():
                continue
            offset = self.offsets.get(role, 0)
            size = path.stat().st_size // 2 * 2
            if size - offset < MIN_CHUNK_SECONDS * BYTES_PER_SECOND:
                continue
            with path.open("rb") as fh:
                fh.seek(offset)
                raw = fh.read(size - offset)
            audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
            chunk_seconds = len(audio) / RATE
            segments = [
                s
                for s in self.model.transcribe(
                    audio,
                    language=None if self.language == "auto" else self.language,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )[0]
                if s.text.strip()
            ]
            consumed = len(raw)
            if segments and chunk_seconds - segments[-1].end < CUT_MARGIN_SECONDS:
                consumed = int(segments[-1].start * RATE) * 2
                segments = segments[:-1]
            start = offset / BYTES_PER_SECOND
            lines += [Line(start + s.start, self.speakers[role], s.text.strip()) for s in segments]
            self.offsets[role] = offset + consumed
        return sorted(lines, key=lambda line: line.start)


@dataclass
class Hint:
    at: float
    kind: str
    text: str
    source: str

    def render(self) -> str:
        m, s = divmod(int(self.at), 60)
        return f"[{m:02d}:{s:02d}] {self.kind.upper()}: {self.text} ({self.source})"


def claude_command(cfg: Config) -> list[str]:
    tools = ",".join(cfg.live.tools)
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(HINT_SCHEMA),
        "--no-session-persistence",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--permission-mode",
        "default",
        "--tools",
        tools,
        "--allowedTools",
        ",".join(cfg.live.allowed_tools),
    ]
    if cfg.live.model:
        cmd += ["--model", cfg.live.model]
    if cfg.live.effort:
        cmd += ["--effort", cfg.live.effort]
    return cmd


def parse_hints(stdout: str) -> list[dict]:
    envelope = json.loads(stdout)
    if envelope.get("is_error"):
        raise RuntimeError(envelope.get("result") or "; ".join(envelope.get("errors", [])) or "claude failed")
    return envelope.get("structured_output", {}).get("hints", [])


@dataclass
class Hinter:
    cfg: Config
    title: str
    me: str
    workdir: Path
    lines: list[Line] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)
    seen: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, lines: list[Line]) -> None:
        with self.lock:
            self.lines += lines

    def due(self) -> bool:
        with self.lock:
            return len(self.lines) > self.seen

    def prompt(self) -> str:
        now = self.lines[-1].start if self.lines else 0.0
        since = now - self.cfg.live.window_minutes * 60
        window = [i for i, line in enumerate(self.lines) if line.start >= since]
        rendered = []
        for i in window:
            if i == self.seen:
                rendered.append(">>> new")
            rendered.append(self.lines[i].render())
        return PROMPT.format(
            name=self.cfg.self_name or "the user",
            me=self.me,
            title=self.title,
            earlier="\n".join(h.render() for h in self.hints) or "(none)",
            transcript="\n".join(rendered),
        )

    def ask(self) -> list[Hint]:
        with self.lock:
            prompt = self.prompt()
            at = self.lines[-1].start
            self.seen = len(self.lines)
        proc = subprocess.run(
            [*claude_command(self.cfg), "--", prompt],
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=HINT_TIMEOUT_S,
        )
        new = [Hint(at, h["kind"], h["text"].strip(), h["source"].strip()) for h in parse_hints(proc.stdout)]
        self.hints += new
        return new


def _notify(hint: Hint) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "-a", "meetingrec", f"Hint: {hint.kind}", hint.text], check=False)


def run(
    directory: Path,
    cfg: Config,
    *,
    title: str,
    roles: list[str],
    stop: threading.Event,
    emit: Emit,
) -> None:
    """Transcribe and hint until `stop` is set, then transcribe what is left."""
    from .transcribe import load_model, resolve_model

    me = cfg.self_name or ("Ich" if cfg.language == "de" else "Me")
    speakers = {"mic": me, "system": "Remote", "room": "Room"}
    model = load_model(cfg, resolve_model(cfg.live.asr_model))
    transcriber = Transcriber(model, directory, roles, cfg.language, speakers)
    workdir = Path(cfg.live.workdir).expanduser() if cfg.live.workdir else Path.cwd()
    hinter = Hinter(cfg, title, me, workdir)
    emit({"event": "live", "workdir": str(workdir), "model": cfg.live.model or "default"})

    def hint_loop() -> None:
        while not stop.wait(cfg.live.hint_interval):
            if not hinter.due():
                continue
            started = time.monotonic()
            try:
                new = hinter.ask()
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                emit({"event": "warning", "kind": "hint-failed", "message": f"hint model failed: {exc}"})
                continue
            with (directory / "hints.md").open("a") as hints_md:
                hints_md.writelines(hint.render() + "\n" for hint in new)
            for hint in new:
                emit(
                    {
                        "event": "hint",
                        "at": round(hint.at, 1),
                        "kind": hint.kind,
                        "text": hint.text,
                        "source": hint.source,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                )
                if cfg.live.notify:
                    _notify(hint)

    threading.Thread(target=hint_loop, daemon=True).start()
    with (directory / "transcript.txt").open("a") as transcript:
        while True:
            stopping = stop.wait(cfg.live.interval)
            lines = transcriber.tick()
            for line in lines:
                transcript.write(line.render() + "\n")
                emit(
                    {
                        "event": "transcript",
                        "at": round(line.start, 1),
                        "speaker": line.speaker,
                        "text": line.text,
                    }
                )
            transcript.flush()
            hinter.add(lines)
            if stopping:
                return


def replay(session_dir: Path, directory: Path, roles: list[str], speed: float, stop: threading.Event) -> None:
    """Feed a recorded meeting's work/<role>.wav into live PCM files at `speed` x real time."""
    step = 0.5
    with ExitStack() as stack:
        sources = {
            role: stack.enter_context(wave.open(str(session_dir / "work" / f"{role}.wav"))) for role in roles
        }
        for role, wav in sources.items():
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
                raise RuntimeError(f"{role}.wav is not 16 kHz mono s16")
        sinks = {role: stack.enter_context((directory / f"{role}.pcm").open("wb")) for role in roles}
        while not stop.is_set():
            fed = False
            for role, wav in sources.items():
                frames = wav.readframes(int(RATE * step * speed))
                if frames:
                    fed = True
                    sinks[role].write(frames)
                    sinks[role].flush()
            if not fed:
                return
            time.sleep(step)
