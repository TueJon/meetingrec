"""Cloud speech recognition: Gemini Transcribe on Google Cloud (Vertex AI), signed in through gcloud.

Each track is cut at quiet moments into chunks of at most CHUNK_SECONDS, because a request
with word timestamps takes at most 15 minutes of audio. Chunks are sent in parallel.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np

from .config import Config
from .model import Segment, TrackRole, Word

CHUNK_SECONDS = 600.0
# A cut goes into the quietest half second of the last CUT_SEARCH seconds before the limit.
CUT_SEARCH = 60.0
CUT_WINDOW = 0.5
SAMPLE_RATE = 16000
# A pause this long, or a sentence end, starts a new segment.
SEGMENT_GAP = 1.0
MAX_VOCABULARY = 1000
PARALLEL_REQUESTS = 8
RETRIES = 3

_GERMAN = {"und", "ich", "das", "ist", "nicht", "wir", "die", "der", "es", "ja", "auch", "aber", "schon"}
_ENGLISH = {"and", "i", "the", "is", "not", "we", "it", "yes", "also", "but", "you", "that", "so"}


def _gcloud(*args: str) -> str:
    out = subprocess.run(["gcloud", *args], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"gcloud {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def project(cfg: Config) -> str:
    found = cfg.asr.cloud_project or _gcloud("config", "get-value", "project")
    if not found:
        raise RuntimeError("no Google Cloud project: set [asr] cloud_project or `gcloud config set project`")
    return found


def endpoint(project_id: str, location: str, model: str) -> str:
    if location == "global":
        host = "aiplatform.googleapis.com"
    elif location in ("eu", "us"):
        host = f"aiplatform.{location}.rep.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project_id}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )


def cut_points(samples: np.ndarray, rate: int = SAMPLE_RATE) -> list[float]:
    """Chunk boundaries in seconds, from 0 to the end, each chunk at most CHUNK_SECONDS long."""
    total = len(samples) / rate
    points = [0.0]
    window = int(CUT_WINDOW * rate)
    while total - points[-1] > CHUNK_SECONDS:
        lo = int((points[-1] + CHUNK_SECONDS - CUT_SEARCH) * rate)
        hi = int((points[-1] + CHUNK_SECONDS) * rate)
        energy = np.convolve(samples[lo:hi].astype(np.float64) ** 2, np.ones(window), mode="valid")
        points.append((lo + int(np.argmin(energy)) + window // 2) / rate)
    points.append(total)
    return points


def _flac(wav: Path, start: float, end: float) -> bytes:
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(wav),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "flac",
            "-f",
            "flac",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout


def _seconds(value: str) -> float:
    return float(value.rstrip("s"))


def _post(url: str, token: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(RETRIES):
        request = urllib.request.Request(
            url, data=data, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code not in (429, 500, 502, 503, 504) or attempt == RETRIES - 1:
                raise RuntimeError(f"Gemini Transcribe HTTP {exc.code}: {detail[:500]}") from None
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def transcribe_chunk(url: str, token: str, audio: bytes, transcription: dict) -> list[Word]:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "audio/flac", "data": base64.b64encode(audio).decode()}}
                ],
            }
        ],
        "generationConfig": {"audioTranscriptionConfig": transcription},
    }
    response = _post(url, token, body)
    parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return [
        Word(_seconds(w["startOffset"]), _seconds(w["endOffset"]), w["word"])
        for part in parts
        for w in part.get("audioTranscription", {}).get("words", [])
    ]


def words_to_segments(words: list[Word], role: TrackRole) -> list[Segment]:
    speaker = "self" if role == "mic" else None
    groups: list[list[Word]] = []
    for word in words:
        if (
            groups
            and word.start - groups[-1][-1].end < SEGMENT_GAP
            and not re.search(r"[.?!]$", groups[-1][-1].text)
        ):
            groups[-1].append(word)
        else:
            groups.append([word])
    return [
        Segment(g[0].start, g[-1].end, " ".join(w.text for w in g), role, speaker=speaker, words=g)
        for g in groups
    ]


def guess_language(segments: list[Segment]) -> str:
    words = [w.lower() for s in segments for w in re.findall(r"\w+", s.text)]
    german = sum(w in _GERMAN for w in words)
    english = sum(w in _ENGLISH for w in words)
    return "de" if german > english else "en"


def transcribe_tracks(
    tracks: dict[TrackRole, tuple[Path, float]], cfg: Config
) -> tuple[dict[TrackRole, list[Segment]], str]:
    """Segments per track role, plus the meeting language."""
    from faster_whisper import decode_audio

    url = endpoint(project(cfg), cfg.asr.cloud_location, cfg.asr.cloud_model)
    token = _gcloud("auth", "print-access-token")
    transcription: dict = {"wordTimestamp": True}
    if cfg.language != "auto":
        transcription["languageCodes"] = [cfg.language]
    if cfg.glossary:
        transcription["customVocabulary"] = cfg.glossary[:MAX_VOCABULARY]

    jobs = []
    for role, (wav, offset) in tracks.items():
        points = cut_points(decode_audio(str(wav), sampling_rate=SAMPLE_RATE))
        jobs += [(role, wav, offset, start, end) for start, end in zip(points, points[1:], strict=False)]
    click.echo(f"[asr] {cfg.asr.cloud_model} ({cfg.asr.cloud_location}): {len(jobs)} chunk(s)")

    def run(job) -> tuple[TrackRole, list[Word]]:
        role, wav, offset, start, end = job
        words = transcribe_chunk(url, token, _flac(wav, start, end), transcription)
        return role, [Word(w.start + start + offset, w.end + start + offset, w.text) for w in words]

    by_role: dict[TrackRole, list[Word]] = {role: [] for role in tracks}
    with ThreadPoolExecutor(PARALLEL_REQUESTS) as pool:
        for role, words in pool.map(run, jobs):
            by_role[role] += words
    segments = {role: words_to_segments(words, role) for role, words in by_role.items()}
    language = (
        cfg.language if cfg.language != "auto" else guess_language([s for v in segments.values() for s in v])
    )
    return segments, language
