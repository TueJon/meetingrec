"""Start the GTK window with the system Python, which carries PyGObject (GTK 4, libadwaita).

PyGObject cannot be pip-installed without the GObject build headers, so the window runs
outside the venv and drives this CLI as a subprocess (`MEETINGREC_CMD`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click

SYSTEM_PYTHON = "/usr/bin/python3"
_PROBE = "; ".join(
    [
        "import gi",
        "gi.require_version('Gtk', '4.0')",
        "gi.require_version('Adw', '1')",
        "from gi.repository import Adw",
    ]
)


def launch() -> None:
    probe = subprocess.run([SYSTEM_PYTHON, "-c", _PROBE], capture_output=True)
    if probe.returncode != 0:
        raise click.ClickException(
            "the window needs GTK 4 and libadwaita for the system Python: "
            "sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1"
        )
    script = Path(__file__).with_name("gui.py")
    env = {**os.environ, "MEETINGREC_CMD": json.dumps([sys.executable, "-m", "meetingrec"])}
    os.execve(SYSTEM_PYTHON, [SYSTEM_PYTHON, "-I", str(script)], env)
