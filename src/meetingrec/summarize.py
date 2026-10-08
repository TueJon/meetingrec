"""Structured, grounded meeting minutes: summary.json + summary.md.

The Claude CLI produces schema-constrained JSON; everything the model could get
wrong mechanically (owners, dates, timestamps, quotes) is validated in plain
Python afterwards and flagged as a warning on the item. Markdown is rendered
from the validated JSON only.
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from . import render
from .config import Config
from .model import Keyframe, Transcript
from .session import Session

SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"

CALENDAR_DAYS_BEFORE = 7
CALENDAR_DAYS_AFTER = 42
QUOTE_WINDOW_S = 120.0
QUOTE_THRESHOLD = 0.8
REF_TOLERANCE_S = 5.0
CLAUDE_TIMEOUT_S = 1200

WEEKDAYS = {
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}
WEEKDAYS_SHORT = {
    "de": ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"],
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}

_REFS = {"type": "array", "items": {"type": "string", "description": "timestamp mm:ss or h:mm:ss"}}


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or list(properties),
        "additionalProperties": False,
    }


_TEXT_ITEM = _obj({"text": {"type": "string"}, "refs": _REFS})
SCHEMA = _obj(
    {
        "summary": {"type": "string", "description": "2-4 sentences"},
        "key_points": {"type": "array", "items": _TEXT_ITEM},
        "decisions": {"type": "array", "items": _TEXT_ITEM},
        "action_items": {
            "type": "array",
            "items": _obj(
                {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "due_phrase": {"type": ["string", "null"], "description": "verbatim"},
                    "due_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
                    "quote": {"type": "string", "description": "short verbatim quote"},
                    "ref": {"type": "string"},
                }
            ),
        },
        "open_questions": {"type": "array", "items": _TEXT_ITEM},
    }
)

LABELS = {
    "de": {
        "summary": "Zusammenfassung", "key_points": "Wichtigste Punkte", "decisions": "Entscheidungen",
        "action_items": "To-dos", "open_questions": "Offene Fragen", "none": "keine",
        "date": "Datum", "time": "Uhrzeit", "duration": "Dauer", "participants": "Teilnehmer",
        "cols": ("Aufgabe", "Verantwortlich", "Frist", "Beleg"),
        "note": "Hinweis: Dieses Protokoll wurde automatisch erstellt. "
                "Das Transkript ist die maßgebliche Quelle.",
        "warnings": "Prüfhinweise", "task": "To-do", "quote_missing": "Zitat nicht im Transkript gefunden",
        "owner_unknown": "Verantwortliche/r „{}“ steht nicht in der Teilnehmerliste",
        "date_invalid": "Datum „{}“ ungültig, liegt vor dem Meeting oder passt nicht zum Wochentag; entfernt",
        "ref_invalid": "Zeitangabe „{}“ liegt außerhalb des Transkripts",
    },
    "en": {
        "summary": "Summary", "key_points": "Key points", "decisions": "Decisions",
        "action_items": "Action items", "open_questions": "Open questions", "none": "none",
        "date": "Date", "time": "Time", "duration": "Duration", "participants": "Participants",
        "cols": ("Task", "Owner", "Due", "Evidence"),
        "note": "Note: these minutes are generated automatically. The transcript is the source of truth.",
        "warnings": "Validation warnings", "task": "Action item",
        "quote_missing": "quote not found in transcript",
        "owner_unknown": "owner “{}” is not in the participant list",
        "date_invalid": "date “{}” is invalid, precedes the meeting or contradicts the weekday; dropped",
        "ref_invalid": "timestamp “{}” is outside the transcript",
    },
}


class SummaryError(RuntimeError):
    pass


@dataclass
class ClaudeResult:
    data: dict
    model: str


# (command, stdin, cwd) -> completed process; replaced in tests.
Runner = Callable[[list[str], str, Path], "subprocess.CompletedProcess[str]"]


def run_claude(cmd: list[str], stdin: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, input=stdin, cwd=cwd, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT_S
        )
    except FileNotFoundError as e:
        raise SummaryError("the `claude` CLI is not installed or not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise SummaryError(f"claude timed out after {CLAUDE_TIMEOUT_S}s") from e


# ---- context ----------------------------------------------------------------


def resolve_language(cfg: Config, transcript: Transcript) -> str:
    lang = cfg.summary.language
    if lang == "auto":
        lang = transcript.language
    return lang if lang in ("de", "en") else "en"


def meeting_start(session: Session, cfg: Config) -> dt.datetime:
    started = session.meta.started
    if cfg.timezone:
        return started.astimezone(ZoneInfo(cfg.timezone))
    return started if started.tzinfo else started.astimezone()


def utc_offset(when: dt.datetime) -> str:
    off = when.utcoffset() or dt.timedelta()
    minutes = int(off.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def calendar_lines(meeting_date: dt.date, lang: str) -> list[str]:
    lines = []
    for offset in range(-CALENDAR_DAYS_BEFORE, CALENDAR_DAYS_AFTER + 1):
        day = meeting_date + dt.timedelta(days=offset)
        line = f"{WEEKDAYS_SHORT[lang][day.weekday()]} {day.isoformat()}"
        lines.append(line + ("  <- meeting day" if offset == 0 else ""))
    return lines


def duration_of(transcript: Transcript) -> float:
    return max((s.end for s in transcript.segments), default=0.0)


def speaker_allow_list(transcript: Transcript, cfg: Config) -> tuple[list[str], list[str]]:
    """(valid owner names, unnamed speaker labels that must not be guessed)."""
    named = [i.name for i in transcript.speakers.values() if i.name]
    allowed = list(dict.fromkeys([*named, *cfg.participants, *([cfg.self_name] if cfg.self_name else [])]))
    unknown = [
        n for n in render.participants(transcript)
        if n not in allowed and (n in ("?", "Remote") or n in transcript.speakers)
    ]
    return allowed, unknown


def build_prompt(
    session: Session, transcript: Transcript, cfg: Config, keyframes: list[Keyframe], lang: str
) -> str:
    start = meeting_start(session, cfg)
    weekday = WEEKDAYS["en"][start.weekday()]
    allowed, unknown = speaker_allow_list(transcript, cfg)
    out_lang = "German" if lang == "de" else "English"
    notes_path = session.path("notes.md")
    notes = notes_path.read_text().strip() if notes_path.exists() else ""

    parts = [
        f"You write structured minutes of a meeting. Write all text in {out_lang}.",
        "",
        "## Rules",
        "- Use only facts from the transcript, the notes and the screen frames. Never invent anything.",
        "- Resolve relative dates (\"next Thursday\", \"übernächste Woche\", \"in two weeks\") ONLY with the "
        "calendar below, counting from the meeting day. Keep the speaker's wording verbatim in due_phrase. "
        "If no date is stated, set due_phrase and due_date to null.",
        "- action_items[].owner must be exactly one name from the participant list, otherwise null. "
        "Unnamed speakers must never be guessed or merged; Whisper often garbles names, so match them "
        "against the participant list and glossary instead of creating new people.",
        "- Every item cites the timestamp(s) (mm:ss or h:mm:ss, as in the transcript) where it was said. "
        "Every action item carries a short verbatim quote from the transcript and its timestamp in ref.",
        f"- If a section has nothing, return an empty list (rendered as \"{LABELS[lang]['none']}\").",
        "- summary: 2 to 4 sentences.",
        "",
        "## Meeting",
        f"- Title: {session.meta.title}",
        f"- Start: {weekday} {start:%Y-%m-%d %H:%M} ({utc_offset(start)})",
        f"- Duration: {render.format_ts(duration_of(transcript))}",
        "",
        "## Calendar",
        *calendar_lines(start.date(), lang),
        "",
        "## Participants (allowed owners)",
        ", ".join(allowed) or "(none known)",
    ]
    if unknown:
        parts += [f"Unnamed speakers (identity unknown, do not guess): {', '.join(unknown)}"]
    if cfg.glossary:
        parts += ["", "## Glossary (correct spellings)", ", ".join(cfg.glossary)]
    if notes:
        parts += [
            "", "## Notes by the participant",
            "Authoritative over the transcript for names, owners and dates.", "", notes,
        ]
    if keyframes:
        parts += [
            "", "## Screen frames",
            "Look at these images (Read tool) for on-screen facts: URLs, names, numbers, slide content.",
            *(f"[{render.format_ts(k.time)}] {k.path}" for k in keyframes),
        ]
    parts += ["", "## Transcript", *render.transcript_lines(transcript)]
    return "\n".join(parts)


# ---- Claude call ------------------------------------------------------------


def claude_command(cfg: Config, session_dir: Path, with_frames: bool) -> list[str]:
    cmd = [
        "claude", "--print", "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
        "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands",
    ]
    if with_frames:
        cmd += ["--tools", "Read", "--allowedTools", "Read", "--add-dir", str(session_dir)]
    else:
        cmd += ["--tools", ""]
    if cfg.summary.model:
        cmd += ["--model", cfg.summary.model]
    if cfg.summary.effort:
        cmd += ["--effort", cfg.summary.effort]
    return cmd


def parse_envelope(proc: subprocess.CompletedProcess[str]) -> ClaudeResult:
    """Extract the structured object from `claude --output-format json` output.

    Success: {"type":"result","subtype":"success","is_error":false,"structured_output":{...},
    "modelUsage":{<model>:...}}. Errors carry is_error:true and `result` or `errors[]`.
    """
    stderr = proc.stderr.strip()
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        env = None
    if not isinstance(env, dict):
        raise SummaryError(f"claude exited {proc.returncode} without JSON output. stderr: {stderr[-800:]}")
    if proc.returncode != 0 or env.get("is_error") or env.get("subtype") != "success":
        detail = env.get("result") or "; ".join(env.get("errors") or []) or env.get("subtype")
        raise SummaryError(f"claude failed (exit {proc.returncode}): {detail}. stderr: {stderr[-800:]}")
    data = env.get("structured_output")
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
        raise SummaryError(f"claude returned no structured output. result: {str(env.get('result'))[:400]}")
    return ClaudeResult(data, next(iter(env.get("modelUsage") or {}), ""))


# ---- validation -------------------------------------------------------------


def parse_ref(ref: str) -> float | None:
    m = re.fullmatch(r"\s*\[?(?:(\d+):)?(\d{1,2}):(\d{2})\]?\s*", str(ref))
    if not m:
        return None
    return int(m[1] or 0) * 3600 + int(m[2]) * 60 + int(m[3])


def match_owner(owner: str | None, allowed: list[str]) -> str | None:
    """Canonical allow-list name for `owner`: exact (case-insensitive) or a unique first-name match."""
    if not owner or not owner.strip():
        return None
    tokens = owner.casefold().split()
    hits = []
    for name in allowed:
        other = name.casefold().split()
        shorter = min(tokens, other, key=len)
        if tokens[: len(shorter)] == other[: len(shorter)]:
            hits.append(name)
    return hits[0] if len(hits) == 1 else None


_WEEKDAY_WORDS = {
    **{w.casefold(): i for i, w in enumerate(WEEKDAYS["de"])},
    **{w.casefold(): i for i, w in enumerate(WEEKDAYS["en"])},
    "mittwochs": 2, "donnerstags": 3, "montags": 0, "dienstags": 1, "freitags": 4,
    "samstags": 5, "sonntags": 6, "sonnabend": 5,
}
_ABBREV = {a: i for lang in WEEKDAYS_SHORT.values() for i, a in enumerate(lang)}


def named_weekdays(phrase: str | None) -> set[int]:
    if not phrase:
        return set()
    found = set()
    for word in re.findall(r"[^\W\d_]+\.?", phrase):
        bare = word.rstrip(".")
        if bare.casefold() in _WEEKDAY_WORDS:
            found.add(_WEEKDAY_WORDS[bare.casefold()])
        elif bare in _ABBREV and (len(bare) == 3 or word.endswith(".") or bare not in ("So", "Do")):
            found.add(_ABBREV[bare])
    return found


def date_problem(due_date: str | None, due_phrase: str | None, meeting_date: dt.date) -> bool:
    """True if due_date is unparsable, before the meeting, or contradicts a weekday in the phrase."""
    if not due_date:
        return False
    try:
        day = dt.date.fromisoformat(due_date)
    except ValueError:
        return True
    weekdays = named_weekdays(due_phrase)
    return day < meeting_date or bool(weekdays and day.weekday() not in weekdays)


def _words(text: str) -> list[str]:
    return re.sub(r"[\W_]+", " ", text.casefold()).split()


def quote_found(quote: str, text: str, threshold: float = QUOTE_THRESHOLD) -> bool:
    q, words = _words(quote), _words(text)
    if not q or not words:
        return False
    needle = " ".join(q)
    if len(words) <= len(q):
        return difflib.SequenceMatcher(None, needle, " ".join(words)).ratio() >= threshold
    for i in range(len(words) - len(q) + 1):
        cand = " ".join(words[i : i + len(q)])
        matcher = difflib.SequenceMatcher(None, needle, cand)
        if matcher.real_quick_ratio() >= threshold and matcher.ratio() >= threshold:
            return True
    return False


def text_near(transcript: Transcript, at: float | None, radius: float = QUOTE_WINDOW_S) -> str:
    segs = transcript.segments
    if at is not None:
        segs = [s for s in segs if s.end >= at - radius and s.start <= at + radius]
    return " ".join(s.text for s in sorted(segs, key=lambda s: s.start))


def validate(
    data: dict, transcript: Transcript, allowed: list[str], meeting_date: dt.date, lang: str
) -> dict:
    """Return a copy of `data` where every item has a `warnings` list; invalid owners/dates are dropped."""
    labels = LABELS[lang]
    limit = duration_of(transcript) + REF_TOLERANCE_S

    def ref_warnings(refs: list[str]) -> list[str]:
        bad = [r for r in refs if (t := parse_ref(r)) is None or t > limit]
        return [labels["ref_invalid"].format(r) for r in bad]

    out = {"summary": data["summary"]}
    for key in ("key_points", "decisions", "open_questions"):
        out[key] = [
            {**item, "refs": list(item.get("refs", [])), "warnings": ref_warnings(item.get("refs", []))}
            for item in data.get(key, [])
        ]
    items = []
    for item in data.get("action_items", []):
        item = {**item, "warnings": ref_warnings([item.get("ref", "")])}
        owner = item.get("owner")
        item["owner"] = match_owner(owner, allowed)
        if owner and item["owner"] is None:
            item["warnings"].append(labels["owner_unknown"].format(owner))
        if date_problem(item.get("due_date"), item.get("due_phrase"), meeting_date):
            item["warnings"].append(labels["date_invalid"].format(item["due_date"]))
            item["due_date"] = None
        at = parse_ref(item.get("ref", ""))
        if not quote_found(item.get("quote", ""), text_near(transcript, at)):
            item["warnings"].append(labels["quote_missing"])
        items.append(item)
    out["action_items"] = items
    return out


# ---- markdown ---------------------------------------------------------------


def _refs(refs: list[str]) -> str:
    return " ".join(f"[{r.strip('[] ')}]" for r in refs)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _bullets(items: list[dict], labels: dict) -> list[str]:
    if not items:
        return [labels["none"]]
    return [
        f"- {i['text']} {_refs(i['refs'])}".rstrip() + (" ⚠️" if i["warnings"] else "")
        for i in items
    ]


def _todo_row(item: dict, lang: str) -> str:
    labels = LABELS[lang]
    due_date = item.get("due_date")
    if due_date:
        day = dt.date.fromisoformat(due_date)
        due = f"{item.get('due_phrase') or ''} → {WEEKDAYS_SHORT[lang][day.weekday()]} {due_date}".strip(" →")
    else:
        due = (item.get("due_phrase") or "—") + " ⚠️"
    owner = item.get("owner") or "? ⚠️"
    quote_bad = labels["quote_missing"] in item["warnings"]
    evidence = f"[{item['ref'].strip('[] ')}] „{item['quote']}“" + (" ⚠️" if quote_bad else "")
    task = item["task"] + (" ⚠️" if item["warnings"] else "")
    return "| " + " | ".join(_cell(c) for c in (task, owner, due, evidence)) + " |"


def collect_warnings(result: dict, lang: str) -> list[str]:
    labels = LABELS[lang]
    found = []
    for key in ("key_points", "decisions", "open_questions"):
        found += [f"{labels[key]} „{i['text'][:60]}“: {w}" for i in result[key] for w in i["warnings"]]
    found += [
        f"{labels['task']} „{i['task'][:60]}“: {w}" for i in result["action_items"] for w in i["warnings"]
    ]
    return found


def render_markdown(
    result: dict, session: Session, transcript: Transcript, cfg: Config, lang: str
) -> str:
    labels = LABELS[lang]
    start = meeting_start(session, cfg)
    weekday = WEEKDAYS[lang][start.weekday()]
    date = f"{start:%d.%m.%Y}" if lang == "de" else f"{start:%Y-%m-%d}"
    head, sep = labels["cols"], "| --- | --- | --- | --- |"
    todos = [f"| {' | '.join(head)} |", sep, *(_todo_row(i, lang) for i in result["action_items"])]
    md = [
        f"# {session.meta.title}",
        "",
        f"- {labels['date']}: {weekday}, {date}",
        f"- {labels['time']}: {start:%H:%M} ({utc_offset(start)})",
        f"- {labels['duration']}: {render.format_ts(duration_of(transcript))}",
        f"- {labels['participants']}: {', '.join(render.participants(transcript)) or '—'}",
        "",
        f"## {labels['summary']}", "", result["summary"], "",
        f"## {labels['key_points']}", "", *_bullets(result["key_points"], labels), "",
        f"## {labels['decisions']}", "", *_bullets(result["decisions"], labels), "",
        f"## {labels['action_items']}", "",
        *(todos if result["action_items"] else [labels["none"]]), "",
        f"## {labels['open_questions']}", "", *_bullets(result["open_questions"], labels), "",
        "---", "", f"_{labels['note']}_",
    ]
    warnings = collect_warnings(result, lang)
    if warnings:
        md += ["", f"**{labels['warnings']}:**", "", *(f"- ⚠️ {w}" for w in warnings)]
    return "\n".join(md) + "\n"


# ---- entry point ------------------------------------------------------------


def summarize_session(
    session: Session,
    transcript: Transcript,
    cfg: Config,
    *,
    keyframes: list[Keyframe],
    runner: Runner = run_claude,
) -> Path:
    lang = resolve_language(cfg, transcript)
    prompt = build_prompt(session, transcript, cfg, keyframes, lang)
    cmd = claude_command(cfg, session.dir, bool(keyframes))
    claude = parse_envelope(runner(cmd, prompt, session.dir))

    allowed, _ = speaker_allow_list(transcript, cfg)
    result = validate(claude.data, transcript, allowed, meeting_start(session, cfg).date(), lang)
    warnings = collect_warnings(result, lang)

    meta = {
        "model": claude.model or cfg.summary.model,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "asr_model": transcript.asr_model,
        "language": lang,
        "keyframes": [k.path for k in keyframes],
    }
    (session.dir / SUMMARY_JSON).write_text(
        json.dumps({"meta": meta, "raw": claude.data, "result": result, "warnings": warnings},
                   ensure_ascii=False, indent=1) + "\n"
    )
    out = session.dir / SUMMARY_MD
    out.write_text(render_markdown(result, session, transcript, cfg, lang))
    print(f"summary written to {out} ({len(warnings)} warnings)")
    return out
