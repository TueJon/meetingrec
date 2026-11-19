"""Live mode: transcribe the call while it runs and let Claude research hints.

The recorder tees every audio track into `live/<role>.pcm` (16 kHz mono s16le) next to
the recording. `Transcriber` reads what is new every `live.interval` seconds and appends
lines to `live/transcript.txt`. `Hinter` looks at the recent transcript every
`live.hint_interval` seconds, sooner after a question, date or number: a quick model
researches draft hints read-only in `live.workdir` (the user's knowledge base), then
`live.verify_model` checks each draft against its source and the newest lines before it
is shown. Hints are printed, sent as `hint` events and appended to `live/hints.md`; notes
that disagree with each other go to `live/record-fixes.md`; every look is logged to
`live/hint-log.jsonl`. `review` compares the hints with the final transcript after the
meeting.
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
# Detecting the language costs as much as transcribing the chunk, so a confident result is
# kept for a while before the next chunk is detected again.
LANGUAGE_CONFIDENCE = 0.8
LANGUAGE_HOLD_SECONDS = 120.0
RESEARCH_TIMEOUT_S = 90
VERIFY_TIMEOUT_S = 90
REVIEW_TIMEOUT_S = 900
KINDS = ["answer", "ask", "mention", "contradiction", "remind"]
CUE = re.compile(
    r"\?|\d|\b(tomorrow|today|tonight|next week|this week|monday|tuesday|wednesday|thursday|friday|"
    r"release[ds]?|deadline|due|ship(s|ped|ping)?|morgen|heute|nächste woche|diese woche|montag|"
    r"dienstag|mittwoch|donnerstag|freitag|frist|bis wann|not sure|need to check|do we have|"
    r"nicht sicher|haben wir)\b",
    re.IGNORECASE,
)
SOURCE_LINE = re.compile(r"^(.+?):(\d+)")
CUE_TIME = re.compile(r"^\[(\d+):(\d\d)\]")
EXCERPT_LINES_BEFORE = 5
EXCERPT_LINES_AFTER = 10
EXCERPT_LINE_CHARS = 1500
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
                    "headline": {"type": "string", "description": "the point itself, at most 8 words"},
                    "text": {"type": "string", "description": "one line of detail, or empty"},
                    "cue": {
                        "type": "string",
                        "description": "the transcript line the hint responds to, copied with its [mm:ss]",
                    },
                    "replaces": {
                        "type": "integer",
                        "description": "number of the earlier hint this one updates, 0 for a new one",
                    },
                    "source": {"type": "string", "description": "file path and line backing the hint"},
                    "source_date": {
                        "type": "string",
                        "description": "date of that entry, YYYY-MM-DD or unknown",
                    },
                    "quote": {"type": "string", "description": "the sentence in the source, copied verbatim"},
                },
                "required": ["kind", "headline", "text", "cue", "replaces", "source", "source_date", "quote"],
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
                    "later": {"type": "boolean", "description": "right, but its topic has not come up yet"},
                    "reason": {"type": "string"},
                    "kind": {"type": "string", "enum": KINDS},
                    "headline": {"type": "string", "description": "the headline as it should be shown"},
                    "text": {"type": "string", "description": "the detail line as it should be shown"},
                },
                "required": ["index", "keep", "later", "reason", "kind", "headline", "text"],
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
Where the notes point to code on this machine, you may read that code too.

First go through the new lines for everything the records could settle: a release or due
date, a ticket or merge request, a number, who does what, a question nobody answered, and
above all what {me} states, confirms or promises. Look each of them up. Answering without
a lookup is only right when the new lines are small talk.
When speakers disagree, guess, or say they must check how something works, that is the most
valuable lookup: search for the answer itself (the rule, table or acceptance criteria that
settles it), not only for the topic.

Draft a hint for {name} only when it helps right now:
- answer: a question was asked that the records answer
- mention: a fact {name} should bring up
- ask: something {name} should ask or clarify before the topic closes
- contradiction: a speaker stated something as a fact and the records say otherwise about
  the same thing. A proposal, a wording idea, a comparison or a guess is not such a claim
- remind: an open item with these people that has not come up yet

No hint is the normal answer. Rules:
- A hint states what the records say. What someone says in the meeting is never a source,
  and a hint never repeats or confirms it as a fact.
- Skip what the newest lines already answer, ask or settle. Notes about this very meeting
  are not a source either.
- Records age. Rely on the newest entry on a topic and give its date as source_date
  (YYYY-MM-DD, or "unknown"). Never present an old plan or design note as the current state.
- A record matches only when it is about the same thing, not when it shares a word: an
  agent, a template or a token can belong to unrelated features.
- Two records that disagree with each other, or a record the meeting has just made
  outdated, are not hints. List them under record_fixes for after the meeting.
- Say what the record is: what runs today, a draft or branch, a requirement, a proposal.
  Keep words like draft, proposed, planned, not merged. Never present a requirement or a
  draft as what runs today.
- A name the speech recognition may have garbled is matched to the closest name in the
  records first. That the records hold nothing is never a hint.
- A hint belongs to what is on the table now. cue is the transcript line that makes it
  relevant, copied with its [mm:ss]. Without such a line there is no hint.
- One card per point. Never repeat or rephrase an earlier hint or a rejected draft. A draft
  that rests on the same record line as an earlier hint is allowed only when it adds
  something new: then replaces is that hint's number and the draft is the full updated
  hint. Otherwise replaces is 0.
- A parked draft was right but too early. Draft it again once its topic comes up.
- {name} reads the hint at a glance while talking. headline is the point itself in at most
  8 words: the answer, the question to ask, the fact. No lead-in such as "Records say".
  text is one line of at most 20 words with the detail that backs it (names, numbers,
  dates), or empty when the headline says it all. Ticket, merge request and flag names only
  when {name} needs them to act. Both in the language the meeting is held in (German
  meeting: German hint). source is the file path and line, quote the sentence there that
  the hint rests on, copied verbatim.

Meeting: {title}

Earlier hints:
{earlier}

Rejected drafts:
{rejected}

Parked drafts:
{parked}

Transcript (lines after ">>> new" are new since your last look):
{transcript}
"""

VERIFY_PROMPT = """You check draft hints before {name} sees them during a meeting that is running
right now ({today}). "{me}" in the transcript is {name}, "Remote" is everyone else; the
transcript comes from fast speech recognition and has errors. A wrong or late hint costs
more than a missing one. Your working directory is {name}'s knowledge base.

Every draft comes with the lines around its source. Check it against them, and open
files only when you need more, for example to look for a newer record. keep is false when:
- the source does not say what the draft claims, or is about a different thing that only
  shares a word with the conversation
- a newer record on the topic says something else
- the draft repeats what someone said in the meeting instead of the records
- the transcript already contains its conclusion, so it comes too late. The topic being
  discussed is not enough: when the speakers assume something else than the record says, or
  are still searching for what the record holds, the draft is on time
- its source is a note about this very meeting
- it would not change what {name} says or does in the next minute
- it rests on the same record line as a hint already shown and adds nothing new
- it presents a requirement, a draft or a branch as what runs today, or its source covers
  another case than the one discussed, and the shown text cannot be corrected to say so
- it says a speaker stated something the transcript does not contain, or only that the
  records hold nothing

A speaker who is unsure ("I think", "I need to check") has not settled a question: a draft
that answers it from the records is on time. later is true when the draft is right but its
topic has not come up yet; it is then held back and offered again.

Otherwise keep is true. kind is the kind to show: contradiction only when a speaker stated
the opposite as a fact about the same thing, else mention or ask. headline (at most 8
words, the point itself, no lead-in) and text (one detail line of at most 20 words, or
empty) are the hint as it should be shown, in the language of the meeting, corrected where
the source is more precise and naming a draft, requirement or proposal as such. reason is
one short sentence. Return one verdict per draft, with its index.

Meeting: {title}

Hints already shown:
{earlier}

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
- live/feedback.jsonl: the user's own thumbs up or down per hint number (may be missing);
  the last line per hint counts, and a vote outranks your judgment of whether it was useful

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


def has_speech(audio) -> bool:
    from faster_whisper.vad import get_speech_timestamps

    return bool(get_speech_timestamps(audio))


@dataclass
class Transcriber:
    """Turns the growing PCM files into transcript lines, one chunk per call.

    `tails` holds each track's last segment while it may still be cut mid-word: the hint
    model reads it right away, the transcript gets it once the next chunk completes it.
    """

    model: object
    directory: Path
    roles: list[str]
    language: str
    speakers: dict[str, str]
    offsets: dict[str, int] = field(default_factory=dict)
    hotwords: str = ""
    speech: Callable[[object], bool] = has_speech
    tails: dict[str, Line] = field(default_factory=dict)
    detected: str = ""
    detected_until: float = 0.0
    heard: float = 0.0

    def _language(self, now: float) -> str | None:
        if self.language != "auto":
            return self.language
        return self.detected if now < self.detected_until else None

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
            self.heard = max(self.heard, size / BYTES_PER_SECOND)
            self.tails.pop(role, None)
            if not self.speech(audio):
                self.offsets[role] = offset + len(raw)
                continue
            start = offset / BYTES_PER_SECOND
            language = self._language(start)
            found, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                hotwords=self.hotwords or None,
            )
            segments = [s for s in found if s.text.strip()]
            if language is None and getattr(info, "language_probability", 0) >= LANGUAGE_CONFIDENCE:
                self.detected, self.detected_until = info.language, start + LANGUAGE_HOLD_SECONDS
            consumed = len(raw)
            if segments and len(audio) / RATE - segments[-1].end < CUT_MARGIN_SECONDS:
                tail = segments.pop()
                consumed = int(tail.start * RATE) * 2
                self.tails[role] = Line(start + tail.start, self.speakers[role], tail.text.strip())
            lines += [Line(start + s.start, self.speakers[role], s.text.strip()) for s in segments]
            self.offsets[role] = offset + consumed
        return sorted(lines, key=lambda line: line.start)

    def tail(self) -> list[Line]:
        return sorted(self.tails.values(), key=lambda line: line.start)


@dataclass
class Hint:
    at: float
    kind: str
    text: str
    source: str
    date: str = ""
    headline: str = ""
    number: int = 0
    replaces: int = 0

    @property
    def cited(self) -> str:
        return f"{self.source}, {self.date}" if self.date and self.date != "unknown" else self.source

    def render(self) -> str:
        m, s = divmod(int(self.at), 60)
        body = " — ".join(part for part in (self.headline, self.text) if part)
        update = f" (updates #{self.replaces})" if self.replaces else ""
        return f"[{m:02d}:{s:02d}] {self.kind.upper()}: {body} ({self.cited}){update}"


@dataclass
class Call:
    output: dict
    text: str
    tools: list[dict]
    cost: float
    seconds: float

    def log(self) -> dict:
        return {"seconds": self.seconds, "cost": round(self.cost, 4), "tools": self.tools}


def excerpt(source: str, workdir: Path) -> str:
    """The lines around a `path:line` source, so the check needs no lookup for them."""
    match = SOURCE_LINE.match(source.strip())
    if not match:
        return ""
    path = Path(match[1]).expanduser()
    try:
        rows = (path if path.is_absolute() else workdir / path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    first = max(int(match[2]) - 1 - EXCERPT_LINES_BEFORE, 0)
    return "\n".join(
        f"   {number}: {row[:EXCERPT_LINE_CHARS]}"
        for number, row in enumerate(rows[first : int(match[2]) + EXCERPT_LINES_AFTER], first + 1)
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
    for path in extra_dirs:
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
    runner: Runner = run_claude
    lines: list[Line] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    parked: list[str] = field(default_factory=list)
    replaced: set[int] = field(default_factory=set)
    fixes: list[str] = field(default_factory=list)
    checking: list[str] = field(default_factory=list)
    tail: list[Line] = field(default_factory=list)
    tail_seen: str = ""
    heard: float = 0.0
    seen: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, lines: list[Line], tail: list[Line] = (), heard: float = 0.0) -> None:
        with self.lock:
            self.lines += lines
            self.tail = list(tail)
            self.heard = heard

    def _tail_text(self) -> str:
        return " ".join(line.text for line in self.tail)

    def due(self, waited: float) -> str:
        """Why a look is due now: `interval`, `cue` (a question, date or number came up) or ``."""
        with self.lock:
            new = self.lines[self.seen :] + (self.tail if self._tail_text() != self.tail_seen else [])
        if not new:
            return ""
        if waited >= self.cfg.live.cue_interval and any(CUE.search(line.text) for line in new):
            return "cue"
        return "interval" if waited >= self.cfg.live.hint_interval else ""

    def _transcript(self, mark: bool) -> str:
        heard = self.lines + self.tail
        since = heard[-1].start - self.cfg.live.window_minutes * 60
        rendered = []
        for i, line in enumerate(heard):
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
        }

    def _earlier(self) -> str:
        shown = [h for h in self.hints if h.number not in self.replaced]
        return "\n".join(f"#{h.number} {h.render()}" for h in shown) or "(none)"

    def prompt(self) -> str:
        return PROMPT.format(
            **self._frame(),
            earlier=self._earlier(),
            rejected="\n".join(f"- {text}" for text in self.rejected + self.checking) or "(none)",
            parked="\n".join(f"- {text}" for text in self.parked) or "(none)",
            transcript=self._transcript(mark=True),
        )

    def verify_prompt(self, drafts: list[dict]) -> str:
        rendered = [
            f"{i}. {d['kind'].upper()}: {_body(d)}\n   source: {d['source']} ({d['source_date']})\n"
            f"   quote: {d['quote']}\n   cue: {d.get('cue', '')}\n"
            f"{excerpt(d['source'], self.workdir or Path.cwd())}"
            for i, d in enumerate(drafts)
        ]
        with self.lock:
            transcript, earlier = self._transcript(mark=False), self._earlier()
        return VERIFY_PROMPT.format(
            **self._frame(), drafts="\n".join(rendered), earlier=earlier, transcript=transcript
        )

    def research(self) -> tuple[list[dict], list[str], dict]:
        """Draft hints for the newest lines: the drafts, new record fixes and the log entry."""
        live = self.cfg.live
        with self.lock:
            prompt = self.prompt()
            at = (self.lines + self.tail)[-1].start
            heard = self.heard
            self.seen, self.tail_seen = len(self.lines), self._tail_text()
        entry: dict = {
            "at": round(at, 1),
            "heard": round(heard, 1),
            "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        try:
            call = self.runner(
                claude_command(self.cfg, live.model, live.effort, HINT_SCHEMA),
                prompt,
                self.workdir,
                RESEARCH_TIMEOUT_S,
            )
            drafts = call.output.get("hints", [])
            found = [
                f"{fix['text'].strip()} ({fix['source'].strip()})"
                for fix in call.output.get("record_fixes", [])
            ]
        except (RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            entry["error"] = str(exc)
            return [], [], entry
        with self.lock:
            fixes = [fix for fix in found if fix not in self.fixes]
            self.fixes += fixes
            self.checking += [_body(draft) for draft in drafts]
        entry |= {"research": call.log(), "drafts": drafts, "record_fixes": fixes}
        return drafts, fixes, entry

    def check(self, drafts: list[dict], entry: dict) -> list[Hint]:
        """Check the drafts against their sources and the newest lines; returns the hints to show."""
        live = self.cfg.live
        kept, dropped, parked = drafts, [], []
        try:
            if drafts and live.verify_model:
                call = self.runner(
                    claude_command(self.cfg, live.verify_model, live.verify_effort, VERIFY_SCHEMA),
                    self.verify_prompt(drafts),
                    self.workdir,
                    VERIFY_TIMEOUT_S,
                )
                verdicts = {v["index"]: v for v in call.output.get("verdicts", [])}
                entry["verify"] = {**call.log(), "verdicts": call.output.get("verdicts", [])}
                kept = [
                    {**draft, **{key: value for key, value in _shown(verdicts[i]).items() if value}}
                    for i, draft in enumerate(drafts)
                    if i in verdicts and verdicts[i]["keep"]
                ]
                lost = [(i, d) for i, d in enumerate(drafts) if not verdicts.get(i, {}).get("keep")]
                parked = [_body(d) for i, d in lost if verdicts.get(i, {}).get("later")]
                dropped = [_body(d) for i, d in lost if not verdicts.get(i, {}).get("later")]
        except (RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            entry["error"] = str(exc)
            kept = []
        with self.lock:
            new = []
            for h in kept:
                replaces = h.get("replaces", 0)
                known = any(hint.number == replaces for hint in self.hints) and replaces not in self.replaced
                hint = Hint(
                    min(_cue_seconds(h.get("cue", ""), entry["at"]), entry["at"]),
                    h["kind"],
                    h["text"].strip(),
                    h["source"].strip(),
                    h["source_date"].strip(),
                    h.get("headline", "").strip(),
                    len(self.hints) + 1,
                    replaces if known else 0,
                )
                if hint.replaces:
                    self.replaced.add(hint.replaces)
                self.hints.append(hint)
                new.append(hint)
            self.checking = [text for text in self.checking if text not in {_body(d) for d in drafts}]
            self.rejected += dropped
            self.parked = [text for text in self.parked + parked if text not in {_body(h) for h in kept}]
        entry["shown"] = [hint.render() for hint in new]
        return new

    def look(self) -> tuple[list[Hint], list[str], dict]:
        """One look in one go: research, then check."""
        drafts, fixes, entry = self.research()
        return ([] if "error" in entry else self.check(drafts, entry)), fixes, entry


def _body(draft: dict) -> str:
    return " — ".join(part.strip() for part in (draft.get("headline", ""), draft["text"]) if part.strip())


def _shown(verdict: dict) -> dict:
    return {key: verdict.get(key, "").strip() for key in ("kind", "headline", "text")}


def _cue_seconds(cue: str, fallback: float) -> float:
    match = CUE_TIME.match(cue.strip())
    return int(match[1]) * 60 + int(match[2]) if match else fallback


def _notify(hint: Hint) -> None:
    if shutil.which("notify-send"):
        subprocess.run(
            ["notify-send", "-a", "meetingrec", f"Hint: {hint.kind}", hint.headline or hint.text],
            check=False,
        )


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

    began = time.monotonic()
    me = cfg.self_name or ("Ich" if cfg.language == "de" else "Me")
    speakers = {"mic": me, "system": "Remote", "room": "Room"}
    model = load_model(cfg, resolve_model(cfg.live.asr_model))
    transcriber = Transcriber(
        model, directory, roles, cfg.language, speakers, hotwords=", ".join(cfg.glossary)
    )
    workdir = Path(cfg.live.workdir).expanduser() if cfg.live.workdir else Path.cwd()
    hinter = Hinter(cfg, title, me, workdir)
    emit(
        {
            "event": "live",
            "workdir": str(workdir),
            "dir": str(directory),
            "model": cfg.live.model or "default",
        }
    )

    writing = threading.Lock()

    def finish(drafts: list[dict], entry: dict, started: float) -> None:
        new = hinter.check(drafts, entry) if "error" not in entry else []
        entry["shown_at"] = round(time.monotonic() - began, 1)
        entry["lag"] = round(entry["shown_at"] - entry["heard"], 1)
        with writing:
            with (directory / "hint-log.jsonl").open("a") as log:
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            with (directory / "hints.md").open("a") as hints_md:
                hints_md.writelines(hint.render() + "\n" for hint in new)
        if "error" in entry:
            emit(
                {"event": "warning", "kind": "hint-failed", "message": f"hint model failed: {entry['error']}"}
            )
        for hint in new:
            emit(
                {
                    "event": "hint",
                    "at": round(hint.at, 1),
                    "id": hint.number,
                    "replaces": hint.replaces,
                    "kind": hint.kind,
                    "headline": hint.headline,
                    "text": hint.text,
                    "source": hint.cited,
                    "file": hint.source,
                    "date": hint.date,
                    "seconds": round(time.monotonic() - started, 1),
                }
            )
            if cfg.live.notify:
                _notify(hint)

    def hint_loop() -> None:
        last = began - cfg.live.hint_interval
        while not stop.wait(1.0):
            trigger = hinter.due(time.monotonic() - last)
            if not trigger:
                continue
            last = time.monotonic()
            drafts, fixes, entry = hinter.research()
            entry |= {"trigger": trigger, "transcript_behind": round(last - began - entry["heard"], 1)}
            if fixes:
                with writing, (directory / "record-fixes.md").open("a") as fixes_md:
                    fixes_md.writelines(f"- {fix}\n" for fix in fixes)
            if drafts:
                threading.Thread(target=finish, args=(drafts, entry, last), daemon=True).start()
            else:
                finish(drafts, entry, last)

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
            hinter.add(lines, transcriber.tail(), transcriber.heard)
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
    )
    call = runner(
        claude_command(cfg, cfg.live.review_model, "medium", extra_dirs=(session_dir,)),
        prompt,
        workdir,
        REVIEW_TIMEOUT_S,
    )
    report = call.text[max(call.text.find("## "), 0) :].strip()
    if not report:
        raise RuntimeError("the hint review came back empty")
    path = directory / "review.md"
    path.write_text(report + "\n")
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
