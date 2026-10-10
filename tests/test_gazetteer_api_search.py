"""Tests for search(): name lookup with disambiguation context."""

from gazetteer import search


def test_exact_match():
    hits = search("Rochester", limit=5)
    assert hits
    assert hits[0]["text"] == "Rochester"
    assert hits[0]["adm0"] == "United States of America"


def test_disambiguation_context():
    hits = search("Rochester", limit=5)
    contexts = {(h["adm0"], h["adm1"]) for h in hits}
    assert len(contexts) > 1  # NY and Minnesota both present
    for h in hits:
        assert {"x", "y", "text", "priority", "adm0", "adm1", "kind", "pop"} <= set(h)


def test_case_insensitive():
    assert search("rochester", limit=1)[0]["text"] == "Rochester"
    assert search("ROCHESTER", limit=1)[0]["text"] == "Rochester"


def test_prefix_beats_substring():
    # "York" is a prefix of York but a substring of New York; the
    # prefix match must rank first.
    hits = search("York", limit=5)
    assert hits[0]["text"] == "York"


def test_diacritic_name_searchable():
    # Local (diacritic) names are matched too.
    hits = search("Zürich", limit=3)
    assert hits and hits[0]["text"] == "Zurich"


def test_limit_respected():
    assert len(search("San", limit=3)) == 3


def test_empty_and_blank_query():
    assert search("") == []
    assert search("   ") == []


def test_no_match():
    assert search("Xyzzyplugh", limit=5) == []


def test_determinism():
    assert search("Springfield", limit=5) == search("Springfield", limit=5)
