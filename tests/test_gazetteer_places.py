"""Tests for the bundled dataset loader."""

import gazetteer
from gazetteer import dataset_info, load_places


def test_record_count():
    places = load_places()
    assert len(places) == 7342


def test_record_fields():
    p = load_places()[0]
    for field in ("name", "lon", "lat", "pop", "scalerank", "featurecla",
                  "adm0", "adm1", "kind", "priority"):
        assert hasattr(p, field), field
    assert isinstance(p.priority, int)
    assert p.kind in ("city", "town", "other")


def test_coordinates_sane():
    for p in load_places():
        assert -180.0 <= p.lon <= 180.0
        assert -90.0 <= p.lat <= 90.0
        assert p.name, "every record needs label text"


def test_dataset_info_provenance():
    info = dataset_info()
    assert info["loaded_records"] == 7342
    assert "natural" in info["dataset"].lower()
    assert "public domain" in info["license"].lower()
    assert info["source"].startswith("https://")


def test_no_network_at_runtime():
    # The loader must not touch the network: block sockets and reload.
    import socket

    real_create = socket.socket

    def boom(*a, **k):
        raise AssertionError("network used at runtime")

    socket.socket = boom
    try:
        load_places.cache_clear()
        assert len(load_places()) == 7342
    finally:
        socket.socket = real_create
        load_places.cache_clear()
