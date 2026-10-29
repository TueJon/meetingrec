import pytest

from meetingrec.capture import x11

XRANDR = """Monitors: 3
 0: +*DisplayPort-8 2560/597x1440/336+2560+0  DisplayPort-8
 1: +DisplayPort-10 2560/597x1440/336+0+0  DisplayPort-10
 2: +DisplayPort-0 1920/509x1080/286+5120+-120  DisplayPort-0
"""

MONITORS = x11.parse_monitors(XRANDR)


def test_parse_monitors():
    monitors = x11.parse_monitors(XRANDR)
    assert [(m.name, m.x, m.y, m.width, m.height) for m in monitors] == [
        ("DisplayPort-8", 2560, 0, 2560, 1440),
        ("DisplayPort-10", 0, 0, 2560, 1440),
        ("DisplayPort-0", 5120, -120, 1920, 1080),
    ]


def test_parse_geometry():
    shell = "WINDOW=46033986\nX=2634\nY=109\nWIDTH=866\nHEIGHT=629\nSCREEN=0\n"
    assert x11.parse_geometry(shell) == (866, 629)


def test_window_source():
    assert x11.window_source(46033986) == [
        "ximagesrc",
        "xid=46033986",
        "use-damage=false",
        "show-pointer=true",
    ]


def test_monitor_source_end_is_inclusive():
    assert x11.monitor_source(MONITORS[0]) == [
        "ximagesrc",
        "startx=2560",
        "starty=0",
        "endx=5119",
        "endy=1439",
        "use-damage=false",
        "show-pointer=true",
    ]


def test_remembered_monitor_wins_if_still_active():
    assert x11.pick_monitor(MONITORS, "DisplayPort-10", "") is MONITORS[1]


def test_unknown_remembered_monitor_falls_back_to_answer():
    assert x11.pick_monitor(MONITORS, "gone", "2") is MONITORS[2]


def test_single_monitor_needs_no_answer():
    assert x11.pick_monitor(MONITORS[:1], None, "") is MONITORS[0]


@pytest.mark.parametrize("answer", ["", "x", "3", "-4"])
def test_invalid_monitor_answer(answer):
    with pytest.raises(RuntimeError, match="invalid monitor"):
        x11.pick_monitor(MONITORS, None, answer)


def test_no_monitors():
    with pytest.raises(RuntimeError, match="no active monitors"):
        x11.pick_monitor([], None, "")
