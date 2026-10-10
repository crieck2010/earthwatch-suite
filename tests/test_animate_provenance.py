"""Tests for provenance sidecars."""

import json
import os

import pytest

from animate.provenance import (PROVENANCE_SCHEMA, sha256_bytes, sha256_file,
                               write_sidecar)


def test_sha256_file(tmp_path):
    p = str(tmp_path / "a.bin")
    with open(p, "wb") as fh:
        fh.write(b"hello")
    import hashlib
    assert sha256_file(p) == hashlib.sha256(b"hello").hexdigest()
    assert sha256_bytes(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_write_sidecar(tmp_path):
    vid = str(tmp_path / "reel.mp4")
    with open(vid, "wb") as fh:
        fh.write(b"fake-video-bytes")
    sidecar = write_sidecar(vid, source="/tmp/manifest.json",
                            source_sha256="abc123", preset="reel", fmt="mp4",
                            fps=30.0, n_frames=48,
                            params={"crf": 18}, encoder="/usr/bin/ffmpeg")
    assert sidecar == vid + ".provenance.json"
    with open(sidecar, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["schema"] == PROVENANCE_SCHEMA
    assert doc["video"] == "reel.mp4"
    assert doc["video_sha256"] == sha256_file(vid)
    assert doc["source"] == "/tmp/manifest.json"
    assert doc["source_sha256"] == "abc123"
    assert doc["preset"] == "reel"
    assert doc["format"] == "mp4"
    assert doc["fps"] == 30.0
    assert doc["n_frames"] == 48
    assert doc["params"] == {"crf": 18}
    assert doc["encoder"] == "/usr/bin/ffmpeg"
    assert doc["created"]


def test_sidecar_missing_video_ok(tmp_path):
    sidecar = write_sidecar(str(tmp_path / "ghost.mp4"), source="x")
    with open(sidecar, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["video_sha256"] == ""
