import numpy as np

from meetingrec.cloud_asr import CHUNK_SECONDS, cut_points, endpoint, guess_language, words_to_segments
from meetingrec.model import Segment, Word


def test_endpoint_hosts():
    assert endpoint("p", "eu", "m").startswith(
        "https://aiplatform.eu.rep.googleapis.com/v1/projects/p/locations/eu/"
    )
    assert endpoint("p", "global", "m").startswith("https://aiplatform.googleapis.com/")
    assert endpoint("p", "europe-west4", "m").startswith("https://europe-west4-aiplatform.googleapis.com/")
    assert endpoint("p", "eu", "m").endswith("/publishers/google/models/m:generateContent")


def test_short_audio_is_one_chunk():
    assert cut_points(np.ones(16000 * 30, dtype=np.float32), 16000) == [0.0, 30.0]


def test_long_audio_is_cut_in_the_quiet_part():
    rate = 100
    samples = np.ones(int(1500 * rate), dtype=np.float32)
    samples[int(570 * rate) : int(572 * rate)] = 0
    points = cut_points(samples, rate)
    assert points[0] == 0.0 and points[-1] == 1500.0
    assert 570 <= points[1] <= 572
    assert all(b - a <= CHUNK_SECONDS for a, b in zip(points, points[1:], strict=False))


def test_words_split_at_pauses_and_sentence_ends():
    words = [
        Word(0.0, 0.3, "Hallo"),
        Word(0.4, 0.8, "zusammen."),
        Word(0.9, 1.2, "Los"),
        Word(3.0, 3.4, "geht's"),
    ]
    segments = words_to_segments(words, "mic")
    assert [s.text for s in segments] == ["Hallo zusammen.", "Los", "geht's"]
    assert segments[0].start == 0.0 and segments[0].end == 0.8 and segments[0].speaker == "self"
    assert words_to_segments(words, "system")[0].speaker is None


def test_language_guess():
    de = [Segment(0, 1, "Ja, das ist auch nicht schlecht, aber wir schauen", "system")]
    en = [Segment(0, 1, "Yes, and I think we also need the export", "system")]
    assert guess_language(de) == "de"
    assert guess_language(en) == "en"
