"""User configuration: ~/.config/meetingrec/config.toml (all keys optional)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "meetingrec"


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "meetingrec"


@dataclass
class CaptureConfig:
    mic: str = ""  # PulseAudio/PipeWire source; "" = current default source
    system: str = ""  # monitor source; "" = monitor of the current default sink
    video_fps: int = 2
    video_encoder: str = "auto"  # auto | nvh264enc | x264enc


@dataclass
class AsrConfig:
    model: str = "large-v3"
    fast_model: str = "large-v3-turbo"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"
    beam_size: int = 5


@dataclass
class DiarizationConfig:
    enabled: bool = True
    model: str = "pyannote/speaker-diarization-community-1"
    device: str = "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass
class SpeakersConfig:
    # Cosine similarity a cluster needs to a stored voiceprint to be auto-named,
    # and the margin it needs over the second-best voiceprint.
    match_threshold: float = 0.55
    match_margin: float = 0.08


@dataclass
class SummaryConfig:
    model: str = ""  # Claude CLI model alias; "" = CLI default
    effort: str = "medium"
    language: str = "auto"  # auto | de | en
    max_keyframes: int = 12


@dataclass
class Config:
    out_root: Path = field(default_factory=lambda: Path.home() / "Recordings" / "meetings")
    self_name: str = ""  # name for the local mic speaker; "" = "Ich"/"Me"
    language: str = "auto"  # ASR language: auto | de | en | ...
    timezone: str = ""  # IANA name; "" = system local time
    glossary: list[str] = field(default_factory=list)  # names + jargon, spelled right
    participants: list[str] = field(default_factory=list)  # usual attendees
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    speakers: SpeakersConfig = field(default_factory=SpeakersConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    # Where this config was loaded from; set by load_config, not a config key.
    source: Path | None = field(default=None, metadata={"internal": True})


def _apply(obj, data: dict, where: str) -> None:
    known = {f.name: f for f in fields(obj) if not f.metadata.get("internal")}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {where}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config section {where}{key} must be a table")
            _apply(current, value, f"{where}{key}.")
        elif isinstance(current, Path):
            setattr(obj, key, Path(value).expanduser())
        else:
            setattr(obj, key, value)


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or config_dir() / "config.toml"
    if path.exists():
        with path.open("rb") as fh:
            _apply(cfg, tomllib.load(fh), "")
        cfg.source = path
    return cfg
