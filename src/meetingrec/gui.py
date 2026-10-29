"""meetingrec window: quick start with the last setup, or a guided setup, then live recording.

Runs under the system Python (PyGObject, GTK 4, libadwaita) and drives the CLI given in
`MEETINGREC_CMD`: `targets --json` to list what can be recorded, `record|full --events` to
record with JSON-lines levels and warnings. Imports nothing from the meetingrec package.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))  # run as a script under `python -I`
from gui_model import (  # noqa: E402
    METER_FLOOR_DB,
    call_candidates,
    db_fraction,
    entry_title,
    fetch_targets,
    load_setup,
    refresh_for_today,
    save_setup,
)

APP_ID = "io.github.tuejon.meetingrec"
_PREVIEW_LEVEL = re.compile(r"rms=\(GValueArray\)< ([^,>]+)")
_STAGE = re.compile(r"^\[(\w+)\]")
# ---- small widget helpers ----------------------------------------------------


def page(title: str, child: Gtk.Widget, tag: str) -> Adw.NavigationPage:
    view = Adw.ToolbarView()
    view.add_top_bar(Adw.HeaderBar())
    content = Adw.PreferencesPage()
    holder = Adw.PreferencesGroup()
    holder.add(child)
    content.add(holder)
    view.set_content(content)
    return Adw.NavigationPage(title=title, child=view, tag=tag)


def column(*children: Gtk.Widget) -> Gtk.Box:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
    for side in ("top", "bottom", "start", "end"):
        getattr(box, f"set_margin_{side}")(18)
    for c in children:
        box.append(c)
    return box


def pill(label: str, style: str | None, on_click) -> Gtk.Button:
    button = Gtk.Button(label=label, halign=Gtk.Align.CENTER)
    button.add_css_class("pill")
    if style:
        button.add_css_class(style)
    # Run after the click gesture ends: pushing a page mid-gesture leaves GTK's pointer state
    # stuck on the old page, and the new page ignores every click.
    button.connect("clicked", lambda *_: GLib.idle_add(lambda: on_click() and False))
    return button


def radio_row(
    title: str, subtitle: str, group: Gtk.CheckButton | None
) -> tuple[Adw.ActionRow, Gtk.CheckButton]:
    check = Gtk.CheckButton(group=group, valign=Gtk.Align.CENTER)
    row = Adw.ActionRow(title=GLib.markup_escape_text(title), subtitle=GLib.markup_escape_text(subtitle))
    row.add_prefix(check)
    row.set_activatable_widget(check)
    return row, check


def nav_buttons(back, next_label: str, on_next) -> Gtk.Box:
    box = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER)
    if back:
        box.append(pill("Back", None, back))
    box.append(pill(next_label, "suggested-action", on_next))
    return box


# ---- level meter --------------------------------------------------------------


class Meter(Gtk.Box):
    def __init__(self, label: str) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.label = Gtk.Label(label=label, xalign=0)
        self.label.add_css_class("caption-heading")
        self.bar = Gtk.LevelBar(min_value=0, max_value=1)
        self.bar.add_offset_value("low", 0.55)
        self.bar.add_offset_value("high", 0.85)
        self.bar.add_offset_value("full", 1.0)
        self.append(self.label)
        self.append(self.bar)

    def set_db(self, db: float) -> None:
        self.bar.set_value(db_fraction(db))


class MicPreview:
    """A throwaway gst pipeline that meters one source while the user picks a microphone."""

    def __init__(self, meter: Meter) -> None:
        self.meter = meter
        self.proc: subprocess.Popen | None = None

    def start(self, source: str) -> None:
        self.stop()
        argv = ["gst-launch-1.0", "-m", "pulsesrc"]
        if source:
            argv.append(f"device={source}")
        argv += ["!", "level", "interval=100000000", "!", "fakesink", "sync=false"]
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            match = _PREVIEW_LEVEL.search(line)
            if match:
                GLib.idle_add(self.meter.set_db, float(match[1]))

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        self.proc = None
        self.meter.set_db(METER_FLOOR_DB)


# ---- the window ---------------------------------------------------------------


class Window(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Meeting recorder", default_width=560, default_height=720)
        self.setup = load_setup()
        self.targets: dict = {}
        self.proc: subprocess.Popen | None = None
        self.state = "idle"  # idle | recording | stopping | processing
        self.seconds = 0
        self.saved_dir: str | None = None
        self.toasts = Adw.ToastOverlay()
        self.nav = Adw.NavigationView()
        self.toasts.set_child(self.nav)
        self.set_content(self.toasts)
        self.preview: MicPreview | None = None
        self.connect("close-request", self._on_close)
        self.nav.add(self._home_page())

    # ---- data

    def _load_targets(self) -> bool:
        try:
            self.targets = fetch_targets()
            return True
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            self.toast(f"Could not list audio sources: {exc}")
            return False

    def toast(self, text: str) -> None:
        self.toasts.add_toast(Adw.Toast(title=GLib.markup_escape_text(text), timeout=6))

    def _windows(self) -> list[dict]:
        return [w for w in self.targets.get("windows", []) if w["pid"] != os.getpid() and w["title"]]

    # ---- home

    def _home_page(self) -> Adw.NavigationPage:
        icon = Gtk.Image(icon_name="audio-input-microphone-symbolic", pixel_size=96, margin_top=12)
        icon.add_css_class("dim-label")
        heading = Gtk.Label(label="Meeting recorder")
        heading.add_css_class("title-1")
        intro = Gtk.Label(
            label="Records your microphone, the call and optionally a window, "
            "then writes a transcript and minutes on this machine.",
            wrap=True,
            max_width_chars=60,
            justify=Gtk.Justification.CENTER,
        )
        group = Adw.PreferencesGroup()
        self.title_row = Adw.EntryRow(title="Title")
        self.title_row.set_text(self.setup.title if self.setup.title != "meeting" else "")
        group.add(self.title_row)
        self.quick_label = Gtk.Label(
            label=self.setup.describe(), wrap=True, max_width_chars=60, justify=Gtk.Justification.CENTER
        )
        self.quick_label.add_css_class("dim-label")
        folder = Gtk.Button(label="Open recordings folder", halign=Gtk.Align.CENTER)
        folder.add_css_class("flat")
        folder.connect("clicked", lambda *_: self._load_targets() and self._open(self.targets["out_root"]))
        body = column(
            icon,
            heading,
            intro,
            group,
            pill("Quick start", "suggested-action", self._quick_start),
            self.quick_label,
            pill("Set up recording…", None, self._wizard),
            folder,
        )
        return page("Meeting recorder", body, "home")

    def _title(self) -> str:
        return self.title_row.get_text().strip() or "meeting"

    def _quick_start(self) -> None:
        if not self._load_targets():
            return
        self.setup.title = self._title()
        self.setup, notes = refresh_for_today(self.setup, self.targets)
        for note in notes:
            self.toast(note)
        self._start()

    # ---- wizard

    def _wizard(self) -> None:
        if not self._load_targets():
            return
        self.setup.title = self._title()
        self.nav.push(self._kind_page())

    def _kind_page(self) -> Adw.NavigationPage:
        group = Adw.PreferencesGroup(title="What kind of meeting?")
        call, call_check = radio_row(
            "Online call", "Teams, Meet, Discord … your mic plus the call's audio", None
        )
        room, room_check = radio_row(
            "In the room",
            "Everyone around one microphone. Nothing the laptop plays is recorded.",
            call_check,
        )
        (room_check if self.setup.room else call_check).set_active(True)
        group.add(call)
        group.add(room)

        def next_step() -> None:
            self.setup.room = room_check.get_active()
            self.nav.push(self._mic_page() if self.setup.room else self._call_page())

        return page("Meeting type", column(group, nav_buttons(None, "Next", next_step)), "kind")

    def _call_page(self) -> Adw.NavigationPage:
        group = Adw.PreferencesGroup(
            title="Where is the call?",
            description="Only this app's audio is recorded, so music or videos elsewhere stay out. "
            "♪ marks apps playing audio right now.",
        )
        refresh = Gtk.Button(
            icon_name="view-refresh-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Refresh"
        )
        refresh.add_css_class("flat")
        group.set_header_suffix(refresh)
        system, system_check = radio_row(
            "All system audio", "Everything the laptop plays, including music", None
        )
        group.add(system)
        rows: list[tuple[Gtk.CheckButton, dict | None]] = [(system_check, None)]
        suggested, others = call_candidates(self._windows(), self.targets.get("apps", []))
        for entry in suggested:
            row, check = radio_row(entry_title(entry), entry["binary"], system_check)
            group.add(row)
            rows.append((check, entry))
        if others:
            more = Adw.ExpanderRow(title="Other apps", subtitle=f"{len(others)} more")
            for entry in others:
                row, check = radio_row(entry_title(entry), entry["binary"], system_check)
                more.add_row(row)
                rows.append((check, entry))
            group.add(more)
        previous = next(
            (
                c
                for c, e in rows
                if e and e["pid"] == self.setup.app_pid and e["title"] == self.setup.app_title
            ),
            None,
        )
        (previous or (rows[1][0] if suggested else system_check)).set_active(True)

        def refresh_page(*_) -> None:
            if self._load_targets():
                self.nav.pop()
                self.nav.push(self._call_page())

        refresh.connect("clicked", refresh_page)

        def next_step() -> None:
            entry = next(e for c, e in rows if c.get_active())
            if entry:
                self.setup.app_pid, self.setup.app_binary = entry["pid"], entry["binary"]
                self.setup.app_title = entry["title"]
                if entry["window"]:
                    self.setup.window_id, self.setup.window_title = entry["window"], entry["title"]
            else:
                self.setup.app_pid, self.setup.app_binary, self.setup.app_title = None, "", ""
            self.nav.push(self._mic_page())

        return page("Call audio", column(group, nav_buttons(self.nav.pop, "Next", next_step)), "call")

    def _mic_page(self) -> Adw.NavigationPage:
        group = Adw.PreferencesGroup(title="Microphone", description="Speak and watch the meter.")
        meter = Meter("Level")
        self.preview = MicPreview(meter)
        mics = self.targets.get("mics", [])
        names = [""] + [m["name"] for m in mics]
        labels = ["System default"] + [
            m["description"] + (" (default)" if m["default"] else "") for m in mics
        ]
        combo = Adw.ComboRow(title="Input", model=Gtk.StringList.new(labels))
        combo.set_selected(names.index(self.setup.mic) if self.setup.mic in names else 0)
        group.add(combo)
        muted = Adw.Banner(title="This microphone is muted", button_label="Unmute")
        group_box = column(muted, group, meter)

        def current() -> str:
            return names[combo.get_selected()]

        def source_name() -> str:
            return current() or next((m["name"] for m in mics if m["default"]), "")

        def update(*_) -> None:
            self.preview.start(current())
            state = subprocess.run(
                ["pactl", "get-source-mute", source_name()], capture_output=True, text=True
            )
            muted.set_revealed(state.stdout.strip().endswith("yes"))

        def unmute(*_) -> None:
            subprocess.run(["pactl", "set-source-mute", source_name(), "0"], capture_output=True)
            update()

        combo.connect("notify::selected", update)
        muted.connect("button-clicked", unmute)
        update()
        self.preview.stop()  # "showing" starts it once the page is on screen

        def next_step() -> None:
            self.setup.mic = current()
            self.nav.push(self._screen_page())

        nav_page = page("Microphone", column(group_box, nav_buttons(self.nav.pop, "Next", next_step)), "mic")
        nav_page.connect("hidden", lambda *_: self.preview.stop())
        nav_page.connect("showing", lambda *_: self.preview.start(current()))
        return nav_page

    def _screen_page(self) -> Adw.NavigationPage:
        x11 = self.targets.get("session") == "x11"
        group = Adw.PreferencesGroup(
            title="Record the screen too?",
            description="Slides and shared screens become keyframes in the minutes. 2 frames per second.",
        )
        off, off_check = radio_row("No screen recording", "Audio only", None)
        window, window_check = radio_row(
            "A window",
            "Only that window, even when covered" if x11 else "Chosen in the system dialog",
            off_check,
        )
        monitor, monitor_check = radio_row(
            "A whole monitor", "Includes notifications and everything else on it", off_check
        )
        for row in (off, window, monitor):
            group.add(row)
        windows = self._windows()
        window_combo = monitor_combo = None
        if x11 and windows:
            window_combo = Adw.ComboRow(
                title="Window", model=Gtk.StringList.new([w["title"] for w in windows])
            )
            ids = [w["id"] for w in windows]
            if self.setup.window_id in ids:
                window_combo.set_selected(ids.index(self.setup.window_id))
            group.add(window_combo)
        monitors = self.targets.get("monitors", [])
        if x11 and monitors:
            labels = [f"{m['name']} ({m['width']}×{m['height']})" for m in monitors]
            monitor_combo = Adw.ComboRow(title="Monitor", model=Gtk.StringList.new(labels))
            names = [m["name"] for m in monitors]
            if self.setup.monitor in names:
                monitor_combo.set_selected(names.index(self.setup.monitor))
            group.add(monitor_combo)
        {"window": window_check, "monitor": monitor_check}.get(self.setup.screen, off_check).set_active(True)

        def sync(*_) -> None:
            if window_combo:
                window_combo.set_sensitive(window_check.get_active())
            if monitor_combo:
                monitor_combo.set_sensitive(monitor_check.get_active())

        for check in (off_check, window_check, monitor_check):
            check.connect("toggled", sync)
        sync()

        def next_step() -> None:
            self.setup.screen = "window" if window_check.get_active() else "off"
            if monitor_check.get_active():
                self.setup.screen = "monitor"
            if window_combo and self.setup.screen == "window":
                chosen = windows[window_combo.get_selected()]
                self.setup.window_id, self.setup.window_title = chosen["id"], chosen["title"]
            if monitor_combo and self.setup.screen == "monitor":
                self.setup.monitor = monitors[monitor_combo.get_selected()]["name"]
            self.nav.push(self._ready_page())

        return page("Screen", column(group, nav_buttons(self.nav.pop, "Next", next_step)), "screen")

    def _ready_page(self) -> Adw.NavigationPage:
        group = Adw.PreferencesGroup(title="Ready")
        title = Adw.EntryRow(title="Title")
        title.set_text(self.setup.title if self.setup.title != "meeting" else "")
        group.add(title)
        group.add(Adw.ActionRow(title="Setup", subtitle=GLib.markup_escape_text(self.setup.describe())))
        after = Adw.PreferencesGroup(title="Afterwards")
        process = Adw.SwitchRow(title="Transcribe and summarize", subtitle="Runs locally when you stop")
        process.set_active(self.setup.process)
        fast = Adw.SwitchRow(title="Faster, less accurate model")
        fast.set_active(self.setup.fast)
        process.bind_property("active", fast, "sensitive", 0)
        fast.set_sensitive(self.setup.process)
        after.add(process)
        after.add(fast)

        def start() -> None:
            self.setup.title = title.get_text().strip() or "meeting"
            self.title_row.set_text(title.get_text().strip())
            self.setup.process, self.setup.fast = process.get_active(), fast.get_active()
            self._start()

        return page(
            "Ready", column(group, after, nav_buttons(self.nav.pop, "Start recording", start)), "ready"
        )

    # ---- recording

    def _start(self) -> None:
        save_setup(self.setup)
        self.quick_label.set_text(self.setup.describe())
        self.seconds = 0
        self.saved_dir = None
        self.proc = subprocess.Popen(
            self.setup.argv(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.state = "recording"
        self.nav.push(self._recording_page())
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        GLib.timeout_add_seconds(1, self._tick)

    def _recording_page(self) -> Adw.NavigationPage:
        self.clock = Gtk.Label(label="0:00")
        self.clock.add_css_class("title-1")
        self.clock.add_css_class("numeric")
        self.status = Gtk.Label(label="Starting …", wrap=True)
        self.status.add_css_class("dim-label")
        self.banner = Adw.Banner()
        self.warnings: dict[str, str] = {}
        self.meters = {}
        meter_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        roles = (
            [("room", "Room microphone")]
            if self.setup.room
            else [("mic", "Your microphone"), ("system", "Call")]
        )
        for role, label in roles:
            self.meters[role] = Meter(label)
            meter_box.append(self.meters[role])
        self.stop_button = pill("Stop", "destructive-action", self._stop)
        self.spinner = Gtk.Spinner(visible=False)
        self.log = Gtk.TextView(editable=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        log_scroller = Gtk.ScrolledWindow(min_content_height=160, child=self.log)
        self.log_expander = Gtk.Expander(label="Details", child=log_scroller)
        self.done_box = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER, visible=False)
        self.new_button = pill("New recording", "flat", self._home)
        self.new_button.set_visible(False)
        body = column(
            self.banner,
            Gtk.Label(label=GLib.markup_escape_text(self.setup.title), css_classes=["title-3"]),
            self.clock,
            self.status,
            meter_box,
            self.stop_button,
            self.spinner,
            self.done_box,
            self.new_button,
            self.log_expander,
        )
        nav_page = page("Recording", body, "recording")
        nav_page.set_can_pop(False)
        return nav_page

    def _tick(self) -> bool:
        if self.state != "recording":
            return False
        self.seconds += 1
        h, rest = divmod(self.seconds, 3600)
        self.clock.set_text(f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}")
        return True

    def _read(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            GLib.idle_add(self._on_line, line.rstrip("\n"))
        GLib.idle_add(self._on_exit, proc.wait())

    def _on_line(self, line: str) -> bool:
        if line.startswith("{"):
            try:
                self._on_event(json.loads(line))
                return False
            except ValueError:
                pass
        buffer = self.log.get_buffer()
        buffer.insert(buffer.get_end_iter(), line + "\n")
        if self.state == "processing" and _STAGE.match(line):
            self.status.set_text(line)
        return False

    def _on_event(self, event: dict) -> None:
        kind = event["event"]
        if kind == "level" and event["role"] in self.meters:
            self.meters[event["role"]].set_db(event["rms"])
        elif kind == "recording":
            self.status.set_text("Recording")
        elif kind == "warning":
            if event["kind"].startswith("mic-dropout") and "mic-muted" in self.warnings:
                return  # the more precise cause is already shown
            self.warnings[event["kind"]] = event["message"]
            self.banner.set_title(GLib.markup_escape_text("\n".join(self.warnings.values())))
            self.banner.set_revealed(True)
        elif kind == "saved":
            self.saved_dir = event["dir"]
        elif kind == "processing":
            self.state = "processing"
            self.status.set_text("Transcribing …")
            self.stop_button.set_visible(False)
            for meter in self.meters.values():
                meter.set_visible(False)
            self.spinner.set_visible(True)
            self.spinner.start()

    def _stop(self) -> None:
        if self.proc and self.state == "recording":
            self.state = "stopping"
            self.status.set_text("Finishing the file …")
            self.stop_button.set_sensitive(False)
            os.killpg(self.proc.pid, signal.SIGINT)

    def _on_exit(self, code: int) -> bool:
        closing = self.state == "closing"
        self.state = "idle"
        if closing:
            self.close()
            return False
        self.spinner.stop()
        self.spinner.set_visible(False)
        self.stop_button.set_visible(False)
        for meter in self.meters.values():
            meter.set_visible(False)
        if code == 0:
            self.status.set_text("Done" if self.setup.process else "Saved")
        else:
            self.status.set_text("Something went wrong; see Details.")
            self.log_expander.set_expanded(True)
        if self.saved_dir:
            folder = Path(self.saved_dir)
            for name, label, style in (
                ("summary.md", "Minutes", "suggested-action"),
                ("transcript.md", "Transcript", None),
            ):
                if (folder / name).exists():
                    self.done_box.append(pill(label, style, lambda p=folder / name: self._open(p)))
            self.done_box.append(pill("Folder", None, lambda: self._open(folder)))
        self.done_box.set_visible(bool(self.saved_dir))
        self.new_button.set_visible(True)
        self.nav.find_page("recording").set_can_pop(True)
        return False

    def _home(self) -> None:
        self.nav.pop_to_tag("home")

    def _open(self, path) -> None:
        Gio.AppInfo.launch_default_for_uri(Path(path).as_uri(), None)

    def _on_close(self, *_) -> bool:
        if self.state == "idle":
            if self.preview:
                self.preview.stop()
            return False
        if self.state in ("closing", "stopping"):
            self.state = "closing"
            return True
        dialog = Adw.AlertDialog(
            heading="Still running",
            body="Stop the recording and finish the file first, then close."
            if self.state == "recording"
            else "The transcript is still being written. Closing cancels it; the recording itself is saved.",
        )
        dialog.add_response("cancel", "Keep running")
        dialog.add_response("close", "Stop and close" if self.state == "recording" else "Close anyway")
        dialog.set_response_appearance("close", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.connect("response", self._on_close_response)
        dialog.present(self)
        return True

    def _on_close_response(self, _dialog, response: str) -> None:
        if response != "close" or not self.proc:
            return
        if self.state == "recording":
            self._stop()
        else:
            os.killpg(self.proc.pid, signal.SIGTERM)
        self.state = "closing"


def main() -> int:
    app = Adw.Application(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)

    def activate(application: Adw.Application) -> None:
        window = application.get_active_window() or Window(application)
        window.present()

    app.connect("activate", activate)
    return app.run(sys.argv[:1])


if __name__ == "__main__":
    sys.exit(main())
