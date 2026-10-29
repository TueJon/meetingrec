from meetingrec.capture import targets

WMCTRL = """0x01e00017  2 18301  host Microsoft Teams (PWA) - Chat | Microsoft Teams
0x04800003  2 232073 host AWS VPN Client
0x00c00004 -1 2000   host Top Bar
0x0480000a  0 99     host
"""


def test_parse_wmctrl_skips_sticky_panels_and_keeps_titles_with_spaces():
    assert targets.parse_wmctrl(WMCTRL) == [
        {"id": 0x01E00017, "pid": 18301, "title": "Microsoft Teams (PWA) - Chat | Microsoft Teams"},
        {"id": 0x04800003, "pid": 232073, "title": "AWS VPN Client"},
        {"id": 0x0480000A, "pid": 99, "title": ""},
    ]


def test_parse_sources_drops_monitors_and_our_capture_bus():
    sources = [
        {"name": "speakers.monitor", "monitor_source": "speakers"},
        {"name": "meetingrec_app_1.monitor", "monitor_source": ""},
        {"name": "usb-mic", "description": "USB", "monitor_source": ""},
        {"name": "laptop-mic", "description": "Internal", "monitor_source": ""},
    ]
    assert targets.parse_sources(sources, "laptop-mic") == [
        {"name": "usb-mic", "description": "USB", "default": False},
        {"name": "laptop-mic", "description": "Internal", "default": True},
    ]
