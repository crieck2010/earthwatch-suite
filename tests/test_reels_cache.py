import json
import os

from reels.cache import Cache


def test_field_store_round_trip(tmp_path):
    c = Cache(str(tmp_path))
    h = "ab" * 32
    assert not c.has_field(h)
    dest = c.store_field(h, '{"times": []}')
    assert c.has_field(h)
    assert c.field_path(h) == dest
    with open(dest) as fh:
        assert json.load(fh) == {"times": []}


def test_frames_markers(tmp_path):
    c = Cache(str(tmp_path))
    h = "cd" * 32
    assert not c.has_frames(h)
    d = c.frames_dir(h)
    assert os.path.isdir(d)
    assert not c.has_frames(h)  # no manifest.json yet
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        fh.write("{}")
    assert c.has_frames(h)


def test_entry_round_trip(tmp_path):
    c = Cache(str(tmp_path))
    h = "ef" * 32
    assert c.read_entry(h) is None
    c.write_entry(h, {"recipe": "x"})
    assert c.read_entry(h) == {"recipe": "x"}


def test_clear_one(tmp_path):
    c = Cache(str(tmp_path))
    h1, h2 = "aa" * 32, "bb" * 32
    c.store_field(h1, "{}")
    c.store_field(h2, "{}")
    removed = c.clear(h1)
    assert removed >= 1
    assert not c.has_field(h1)
    assert c.has_field(h2)


def test_clear_all(tmp_path):
    c = Cache(str(tmp_path))
    c.store_field("aa" * 32, "{}")
    c.write_entry("bb" * 32, {})
    assert c.clear() >= 2
    assert not c.has_field("aa" * 32)
    assert c.read_entry("bb" * 32) is None


def test_content_hash(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    import hashlib

    assert Cache.content_hash(str(p)) == hashlib.sha256(b"hello").hexdigest()
