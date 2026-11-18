"""Live mode: transcribe the call while it runs and let Claude research hints.

The recorder tees every audio track into `live/<role>.pcm` (16 kHz mono s16le) next to
the recording. `Transcriber` reads what is new every `live.interval` seconds and appends
lines to `live/transcript.txt`. `Hinter` looks at the recent transcript every
`live.hint_interval` seconds, sooner after a question, date or number: a quick model
researches draft hints read-only in `live.workdir` (the user's knowledge base) and
`live.code_dirs`, then `live.verify_model` checks each draft against its source and the
newest lines before it is shown. Hints are printed, sent as `hint` events and appended to
`live/hints.md`; notes that disagree with each other go to `live/record-fixes.md`; every
look is logged to `live/hint-log.jsonl`. `review` compares the hints with the final
transcript after the meeting.
"""

from __future__ import annotations

import datetime as dt
import json
import re
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
RESEARCH_TIMEOUT_S = 90
VERIFY_TIMEOUT_S = 90
REVIEW_TIMEOUT_S = 900
KINDS = ["answer", "ask", "mention", "contradiction", "remind"]
CUE = re.compile(
    r"\?|\d|\b(tomorrow|today|tonight|next week|this week|monday|tuesday|wednesday|thursday|friday|"
    r"release[ds]?|deadline|due|ship(s|ped|ping)?|morgen|heute|nächste woche|diese woche|montag|"
    r"dienstag|mittwoch|donnerstag|freitag|frist|bis wann)\b",
    re.IGNORECASE,
)
LOGGED_TOOL_INPUT = ("file_path", "pattern", "path", "glob", "offset", "limit")

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
                    "source": {"type": "string", "description": "file path and line backing the hint"},
                    "source_date": {
                        "type": "string",
                        "description": "date of that entry, YYYY-MM-DD or unknown",
                    },
                    "quote": {"type": "string", "description": "the sentence in the source, copied verbatim"},
                },
                "required": ["kind", "text", "source", "source_date", "quote"],
                "additionalProperties": False,
            },
        },
        "record_fixes": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "source": {"type": "string"}},
                "required": ["text", "source"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hints", "record_fixes"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "text": {"type": "string", "description": "the hint as it should be shown"},
                },
                "required": ["index", "keep", "reason", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

PROMPT = """You are {name}'s silent assistant during a meeting that is running right now ({today}).
Below is the live transcript of the last minutes: "{me}" is {name}, "Remote" is everyone
else on the call. The transcript comes from fast speech recognition and has errors.
Your working directory is {name}'s knowledge base (notes, open threads, history). Use your
tools to look things up when the conversation touches something that may be recorded there.
{code}
First go through the new lines for everything the records could settle: a release or due
date, a ticket or merge request, a number, who does what, a question nobody answered, and
above all what {me} states, confirms or promises. Look each of them up. Answering without
a lookup is only right when the new lines are small talk.

Draft a hint for {name} only when it helps right now:
- answer: a question was asked that the records answer
- mention: a fact {name} should bring up
- ask: something {name} should ask or clarify before the topic closes
- contradiction: something said contradicts the records
- remind: an open item with these people that has not come up yet

No hint is the normal answer. Rules:
- A hint states what the records say. What someone says in the meeting is never a source,
  and a hint never repeats or confirms it as a fact.
- Skip what the newest lines already answer, ask or settle.
- Records age. Rely on the newest entry on a topic and give its date as source_date
  (YYYY-MM-DD, or "unknown"). Never present an old plan or design note as the current state.
- A record matches only when it is about the same thing, not when it shares a word: an
  agent, a template or a token can belong to unrelated features.
- Two records that disagree with each other, or a record the meeting has just made
  outdated, are not hints. List them under record_fixes for after the meeting.
- Never repeat or rephrase an earlier hint or a rejected draft.
- Each hint at most 25 words, in the language the meeting is held in (German meeting:
  German hint), concrete (names, numbers, dates). source is the file path and line, quote
  the sentence there that the hint rests on, copied verbatim.

Meeting: {title}

Earlier hints:
{earlier}

Rejected drafts:
{rejected}

Transcript (lines after ">>> new" are new since your last look):
{transcript}
"""

VERIFY_PROMPT = """You check draft hints before {name} sees them during a meeting that is running
right now ({today}). "{me}" in the transcript is {name}, "Remote" is everyone else; the
transcript comes from fast speech recognition and has errors. A wrong or late hint costs
more than a missing one. Your working directory is {name}'s knowledge base.
{code}
Open the source of every draft and check it. keep is false when:
- the source does not say what the draft claims, or is about a different thing that only
  shares a word with the conversation
- a newer record on the topic says something else
- the draft repeats what someone said in the meeting instead of the records
- the newest transcript lines already answer, ask or settle it, so it comes too late
- it would not change what {name} says or does in the next minute

Otherwise keep is true. text is the hint as it should be shown: at most 25 words, in the
language of the meeting, corrected where the source is more precise. reason is one short
sentence. Return one verdict per draft, with its index.

Meeting: {title}

Drafts:
{drafts}

Transcript (newest lines last):
{transcript}
"""

REVIEW_PROMPT = """You review how the live hints did in the meeting "{title}" ({today}), so the next
meeting gets better ones. Your working directory is the knowledge base the hints were
researched in. Read these files in {directory}:
- {transcript}: the final transcript, the source of truth for what was said and when
- live/hints.md: the hints that were shown, with the transcript time they refer to
- live/hint-log.jsonl: every look of the hint model: drafts, verdicts, tool calls, seconds, lag
- live/record-fixes.md: notes the hint model found to disagree (may be missing)
{code}
Check the claims yourself against the records and code; do not trust the hints. Write
Markdown with exactly these sections, short and concrete:

## Verdict
Three lines: how many hints were useful, redundant, late or wrong, and the main cause.

## Hints
A table: time, hint, verdict (useful / redundant / late / wrong), why in one sentence.

## Dropped drafts
Drafts the check rejected that should have been shown, or "none".

## Missed moments
Where the records held something that would have helped: transcript time, what was said,
what the records say, path of the record.

## Record fixes
Notes that are outdated or contradict each other after this meeting, with path and line.

## Changes to try
Prompt or settings changes that would have prevented the failures above.
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
    hotwords: str = ""

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
                    hotwords=self.hotwords or None,
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
    date: str = ""

    @property
    def cited(self) -> str:
        return f"{self.source}, {self.date}" if self.date and self.date != "unknown" else self.source

    def render(self) -> str:
        m, s = divmod(int(self.at), 60)
        return f"[{m:02d}:{s:02d}] {self.kind.upper()}: {self.text} ({self.cited})"


@dataclass
class Call:
    output: dict
    text: str
    tools: list[dict]
    cost: float
    seconds: float

    def log(self) -> dict:
        return {"seconds": self.seconds, "cost": round(self.cost, 4), "tools": self.tools}


def code_dirs(cfg: Config) -> list[Path]:
    return [Path(d).expanduser() for d in cfg.live.code_dirs]


def code_section(cfg: Config) -> str:
    rows = []
    for path in code_dirs(cfg):
        branch = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        rows.append(f"- {path}" + (f" (branch {branch})" if branch else ""))
    if not rows:
        return ""
    return (
        "\nFor questions about how the product works today you may also read this code. It is a\n"
        "checkout, so its branch may differ from what is in production:\n" + "\n".join(rows) + "\n"
    )


def claude_command(
    cfg: Config, model: str, effort: str, schema: dict | None = None, extra_dirs: tuple[Path, ...] = ()
) -> list[str]:
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--setting-sources",
        "",
        "--permission-mode",
        "default",
        "--tools",
        ",".join(cfg.live.tools),
        "--allowedTools",
        ",".join(cfg.live.allowed_tools),
    ]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    for path in (*code_dirs(cfg), *extra_dirs):
        cmd += ["--add-dir", str(path)]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def parse_stream(stdout: str) -> tuple[dict, list[dict]]:
    """The result envelope and the tool calls from `claude --output-format stream-json`."""
    tools, result = [], None
    for raw in stdout.splitlines():
        if not raw.startswith("{"):
            continue
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") != "StructuredOutput":
                    used = {k: v for k, v in block.get("input", {}).items() if k in LOGGED_TOOL_INPUT}
                    tools.append({"tool": block["name"], **used})
        elif event.get("type") == "result":
            result = event
    if result is None:
        raise RuntimeError("claude returned no result")
    if result.get("is_error"):
        raise RuntimeError(result.get("result") or "; ".join(result.get("errors", [])) or "claude failed")
    return result, tools


def run_claude(cmd: list[str], prompt: str, cwd: Path, timeout: float) -> Call:
    started = time.monotonic()
    proc = subprocess.run([*cmd, "--", prompt], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    result, tools = parse_stream(proc.stdout)
    return Call(
        result.get("structured_output") or {},
        result.get("result") or "",
        tools,
        result.get("total_cost_usd") or 0.0,
        round(time.monotonic() - started, 1),
    )


Runner = Callable[[list[str], str, Path, float], Call]


@dataclass
class Hinter:
    cfg: Config
    title: str
    me: str
    workdir: Path
    code: str = ""
    runner: Runner = run_claude
    lines: list[Line] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    seen: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, lines: list[Line]) -> None:
        with self.lock:
            self.lines += lines

    def due(self, waited: float) -> str:
        """Why a look is due now: `interval`, `cue` (a question, date or number came up) or ``."""
        with self.lock:
            new = self.lines[self.seen :]
        if not new:
            return ""
        if waited >= self.cfg.live.cue_interval and any(CUE.search(line.text) for line in new):
            return "cue"
        return "interval" if waited >= self.cfg.live.hint_interval else ""

    def _transcript(self, mark: bool) -> str:
        since = self.lines[-1].start - self.cfg.live.window_minutes * 60
        rendered = []
        for i, line in enumerate(self.lines):
            if line.start < since:
                continue
            if mark and i == self.seen:
                rendered.append(">>> new")
            rendered.append(line.render())
        return "\n".join(rendered)

    def _frame(self) -> dict:
        return {
            "name": self.cfg.self_name or "the user",
            "me": self.me,
            "title": self.title,
            "today": dt.date.today().strftime("%A, %Y-%m-%d"),
            "code": self.code,
        }

    def prompt(self) -> str:
        return PROMPT.format(
            **self._frame(),
            earlier="\n".join(h.render() for h in self.hints) or "(none)",
            rejected="\n".join(f"- {text}" for text in self.rejected) or "(none)",
            transcript=self._transcript(mark=True),
        )

    def verify_prompt(self, drafts: list[dict]) -> str:
        rendered = [
            f"{i}. {d['kind'].upper()}: {d['text']}\n   source: {d['source']} ({d['source_date']})\n"
            f"   quote: {d['quote']}"
            for i, d in enumerate(drafts)
        ]
        with self.lock:
            transcript = self._transcript(mark=False)
        return VERIFY_PROMPT.format(**self._frame(), drafts="\n".join(rendered), transcript=transcript)

    def look(self) -> tuple[list[Hint], list[str], dict]:
        """One look: research drafts, check them, and return shown hints, record fixes and the log entry."""
        live = self.cfg.live
        with self.lock:
            prompt = self.prompt()
            at = self.lines[-1].start
            self.seen = len(self.lines)
        entry: dict = {
            "at": round(at, 1),
            "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        try:
            research = self.runner(
                claude_command(self.cfg, live.model, live.effort, HINT_SCHEMA),
                prompt,
                self.workdir,
                RESEARCH_TIMEOUT_S,
            )
            drafts = research.output.get("hints", [])
            entry |= {"research": research.log(), "drafts": drafts}
            fixes = [
                f"{fix['text'].strip()} ({fix['source'].strip()})"
                for fix in research.output.get("record_fixes", [])
            ]
            fixes = [fix for fix in fixes if fix not in self.fixes]
            self.fixes += fixes
            kept = drafts
            if drafts and live.verify_model:
                check = self.runner(
                    claude_command(self.cfg, live.verify_model, live.verify_effort, VERIFY_SCHEMA),
                    self.verify_prompt(drafts),
                    self.workdir,
                    VERIFY_TIMEOUT_S,
                )
                verdicts = {v["index"]: v for v in check.output.get("verdicts", [])}
                entry["verify"] = {**check.log(), "verdicts": check.output.get("verdicts", [])}
                kept = []
                for i, draft in enumerate(drafts):
                    verdict = verdicts.get(i)
                    if verdict and verdict["keep"]:
                        kept.append({**draft, "text": verdict["text"].strip() or draft["text"]})
                    else:
                        self.rejected.append(draft["text"].strip())
        except (RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            entry["error"] = str(exc)
            return [], [], entry
        new = [
            Hint(at, h["kind"], h["text"].strip(), h["source"].strip(), h["source_date"].strip())
            for h in kept
        ]
        self.hints += new
        entry |= {"shown": [h.render() for h in new], "record_fixes": fixes}
        return new, fixes, entry


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
    transcriber = Transcriber(
        model, directory, roles, cfg.language, speakers, hotwords=", ".join(cfg.glossary)
    )
    workdir = Path(cfg.live.workdir).expanduser() if cfg.live.workdir else Path.cwd()
    hinter = Hinter(cfg, title, me, workdir, code_section(cfg))
    emit({"event": "live", "workdir": str(workdir), "model": cfg.live.model or "default"})
    began = time.monotonic()

    def hint_loop() -> None:
        last = began - cfg.live.hint_interval
        while not stop.wait(1.0):
            trigger = hinter.due(time.monotonic() - last)
            if not trigger:
                continue
            last = time.monotonic()
            new, fixes, entry = hinter.look()
            entry |= {"trigger": trigger, "lag": round(time.monotonic() - began - entry["at"], 1)}
            with (directory / "hint-log.jsonl").open("a") as log:
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if "error" in entry:
                emit(
                    {
                        "event": "warning",
                        "kind": "hint-failed",
                        "message": f"hint model failed: {entry['error']}",
                    }
                )
                continue
            if fixes:
                with (directory / "record-fixes.md").open("a") as fixes_md:
                    fixes_md.writelines(f"- {fix}\n" for fix in fixes)
            with (directory / "hints.md").open("a") as hints_md:
                hints_md.writelines(hint.render() + "\n" for hint in new)
            for hint in new:
                emit(
                    {
                        "event": "hint",
                        "at": round(hint.at, 1),
                        "kind": hint.kind,
                        "text": hint.text,
                        "source": hint.cited,
                        "seconds": round(time.monotonic() - last, 1),
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


def review(session_dir: Path, cfg: Config, title: str, runner: Runner = run_claude) -> Path | None:
    """Compare the hints of a finished meeting with its final transcript; writes `live/review.md`."""
    directory = live_dir(session_dir)
    if not cfg.live.review_model or not (directory / "hint-log.jsonl").exists():
        return None
    workdir = Path(cfg.live.workdir).expanduser() if cfg.live.workdir else Path.cwd()
    prompt = REVIEW_PROMPT.format(
        title=title,
        today=dt.date.today().isoformat(),
        directory=session_dir,
        transcript="transcript.md" if (session_dir / "transcript.md").exists() else "live/transcript.txt",
        code=code_section(cfg),
    )
    call = runner(
        claude_command(cfg, cfg.live.review_model, "medium", extra_dirs=(session_dir,)),
        prompt,
        workdir,
        REVIEW_TIMEOUT_S,
    )
    if not call.text.strip():
        raise RuntimeError("the hint review came back empty")
    path = directory / "review.md"
    path.write_text(call.text.strip() + "\n")
    return path


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
