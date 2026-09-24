from pathlib import Path

import pytest

from face_tracker.config import load_config

DEMO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"


def test_loads_nested_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("camera:\n  source: 0\n  mirror: false\n")
    assert load_config(path) == {"camera": {"source": 0, "mirror": False}}


def test_accepts_string_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("answer: 42\n")
    assert load_config(str(path)) == {"answer": 42}


def test_empty_file_returns_empty_dict(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path) == {}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_demo_config_has_valid_camera_section():
    camera = load_config(DEMO_CONFIG)["camera"]
    assert isinstance(camera["source"], int)
    for key in ("width", "height"):
        value = camera.get(key)
        assert value is None or (isinstance(value, int) and value > 0)
    assert isinstance(camera["mirror"], bool)
