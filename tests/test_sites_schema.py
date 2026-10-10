"""Schema file integrity + validator conformance tests."""

import json
import os

import pytest

from sites.scaffold import scaffold
from sites.validate import SCHEMA_VERSION, validate_config

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema",
                            "site-config.schema.json")


def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_schema_file_is_valid_json():
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_schema_version_consistent():
    schema = load_schema()
    assert schema["version"] == SCHEMA_VERSION == "1.0.0"
    assert schema["$id"].endswith("/site-config/v1")


def test_schema_requires_core_fields():
    schema = load_schema()
    required = set(schema["required"])
    assert {"site_id", "name", "geometry", "indices", "thresholds"} <= required


def test_schema_site_id_pattern():
    schema = load_schema()
    pattern = schema["properties"]["site_id"]["pattern"]
    import re
    assert re.match(pattern, "black-rock-quarry")
    assert not re.match(pattern, "Black Rock!")


def test_validator_agrees_with_schema_on_minimal_config():
    """A config with exactly the schema's required fields must validate."""
    config = {
        "site_id": "my-site",
        "name": "My Site",
        "geometry": {"bbox": [-77.7, 43.05, -77.6, 43.12]},
        "indices": ["ndvi"],
        "thresholds": {
            "ndvi": {"warning": {"absolute_delta": 0.1}}
        },
    }
    assert validate_config(config) == []


@pytest.mark.parametrize("template", ["minimal", "quarry", "farm", "coast", "forest"])
def test_all_templates_validate(template):
    config = scaffold(f"test-{template}", f"Test {template}", template=template)
    errors = validate_config(config)
    assert errors == [], f"template {template}: {errors}"


def test_each_required_field_missing_fails():
    base = scaffold("req-test", "Req Test", template="minimal")
    for field in ("site_id", "name", "geometry", "indices", "thresholds"):
        cfg = {k: v for k, v in base.items() if k != field}
        errors = validate_config(cfg)
        assert errors, f"removing {field} should fail validation"
        assert any(field in e for e in errors), f"error should name {field}: {errors}"
