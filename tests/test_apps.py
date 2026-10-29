from meetingrec.capture import apps
from meetingrec.capture.apps import AppSelector, Port, Stream


def _node(node_id, media_class, **props):
    return {
        "id": node_id,
        "type": "PipeWire:Interface:Node",
        "info": {"props": {"media.class": media_class, **props}},
    }


def _port(port_id, node_id, direction, channel):
    props = {"node.id": node_id, "port.direction": direction, "audio.channel": channel}
    return {"id": port_id, "type": "PipeWire:Interface:Port", "info": {"props": props}}


DUMP = [
    {
        "id": 30,
        "type": "PipeWire:Interface:Client",
        "info": {"props": {"application.process.id": 500, "application.process.binary": "pw-play"}},
    },
    _node(
        10,
        "Stream/Output/Audio",
        **{
            "application.name": "Google Chrome",
            "application.process.id": 4321,
            "application.process.binary": "chrome",
            "media.name": "Playback",
        },
    ),
    _node(
        11, "Stream/Output/Audio", **{"application.name": "pw-play", "client.id": 30, "media.name": "x.wav"}
    ),
    _node(12, "Audio/Sink", **{"node.name": "speakers"}),
    _node(13, "Stream/Input/Audio", **{"application.process.id": 4321}),
    _port(100, 10, "out", "FL"),
    _port(101, 10, "out", "FR"),
    {"id": 102, "type": "PipeWire:Interface:Port", "info": {"props": {}}},
]


def test_parse_dump_keeps_playback_streams_and_fills_native_clients_from_the_client():
    streams, ports = apps.parse_dump(DUMP)
    assert streams == [
        Stream(10, "Google Chrome", "chrome", 4321, "Playback"),
        Stream(11, "pw-play", "pw-play", 500, "x.wav"),
    ]
    assert [p.port_id for p in ports] == [100, 101]


def test_matches_by_process_tree_not_by_a_shared_binary_name():
    chrome_audio = Stream(10, "Google Chrome", "chrome", 4321, "")
    ancestors = {4321: [4321, 4000, 1000]}
    assert apps.matches(chrome_audio, AppSelector(pid=4000, binary="chrome"), ancestors)
    # Another Chrome channel (e.g. Chrome Beta playing music) shares the binary but not the tree.
    assert not apps.matches(chrome_audio, AppSelector(pid=7000, binary="chrome"), ancestors)


def test_matches_by_binary_when_a_pid_is_unknown():
    sandboxed = Stream(10, "Discord", "Discord", None, "")
    assert apps.matches(sandboxed, AppSelector(pid=4000, binary="Discord"), {})
    assert apps.matches(Stream(10, "", "Discord", 9, ""), AppSelector(binary="Discord"), {})
    assert not apps.matches(sandboxed, AppSelector(pid=4000, binary="chrome"), {})


def test_link_pairs_by_channel_and_mono_to_both():
    bus = [Port(200, 2, "in", "FL"), Port(201, 2, "in", "FR"), Port(202, 2, "out", "FL")]
    stereo = [Port(100, 1, "out", "FL"), Port(101, 1, "out", "FR"), Port(103, 1, "in", "FL")]
    assert apps.link_pairs(stereo, bus) == [(100, 200), (101, 201)]
    assert apps.link_pairs([Port(105, 1, "out", "MONO")], bus) == [(105, 200), (105, 201)]


def test_selector_parse(monkeypatch):
    monkeypatch.setattr(apps, "process_binary", lambda pid: "chrome")
    assert AppSelector.parse("4000") == AppSelector(pid=4000, binary="chrome")
    assert AppSelector.parse("Discord") == AppSelector(binary="Discord")
    assert AppSelector(pid=4000, binary="chrome").label() == "chrome (pid 4000)"
