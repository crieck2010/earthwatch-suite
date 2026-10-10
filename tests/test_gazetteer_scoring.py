"""Tests for the priority formula and kind classification."""

from gazetteer.scoring import kind_of, priority_of, sort_key


def test_capital_bonus():
    # Same population and scalerank: the capital outranks the plain town.
    town = priority_of(50_000, 7, "Populated place")
    capital = priority_of(50_000, 7, "Admin-1 capital")
    assert capital == town + 10


def test_population_dominates_log_scale():
    small = priority_of(1_000, 8, "Populated place")
    large = priority_of(1_000_000, 8, "Populated place")
    assert large > small
    # log scale: 1000x population is ~30 points, not 1000x the score
    assert large - small == 30


def test_scalerank_penalty():
    # Lower scalerank (more important) scores higher, all else equal.
    assert priority_of(100_000, 2, "Populated place") > priority_of(
        100_000, 9, "Populated place"
    )


def test_priority_is_int():
    assert isinstance(priority_of(12345, 6, "Populated place"), int)


def test_kind_capital():
    assert kind_of("Admin-0 capital", 1_000) == "city"
    assert kind_of("Admin-1 region capital", 1_000) == "city"


def test_kind_by_population():
    assert kind_of("Populated place", 100_000) == "city"
    assert kind_of("Populated place", 99_999) == "town"


def test_kind_other():
    assert kind_of("Scientific station", 5) == "other"
    assert kind_of("Historic place", 500) == "other"


def test_sort_key_total_order():
    # (-priority, name): deterministic, higher priority first.
    class P:
        def __init__(self, name, priority):
            self.name = name
            self.priority = priority

    a, b, c = P("Bravo", 40), P("Alpha", 40), P("Charlie", 50)
    assert sorted([a, b, c], key=sort_key) == [c, b, a]
