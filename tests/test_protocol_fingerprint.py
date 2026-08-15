from pathlib import Path

import pytest

from uav_isac.coordination.protocol_fingerprint import (
    fingerprint_controller_implementation,
    fingerprint_protocol_implementation,
)


def test_fingerprint_covers_source_bytes_and_complete_config_chain(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "cfg").mkdir()
    (tmp_path / "src" / "wire.py").write_text("WIRE=1\n", encoding="utf-8")
    (tmp_path / "cfg" / "base.yaml").write_text(
        "radio:\n  bandwidth: 1\n", encoding="utf-8")
    (tmp_path / "cfg" / "child.yaml").write_text(
        "extends: base.yaml\nradio:\n  power: 2\n", encoding="utf-8")
    kwargs = dict(
        workspace_root=tmp_path,
        implementation_paths=("src/wire.py",),
    )
    first = fingerprint_protocol_implementation(
        tmp_path / "cfg" / "child.yaml", **kwargs)
    repeated = fingerprint_protocol_implementation(
        tmp_path / "cfg" / "child.yaml", **kwargs)
    assert first == repeated
    assert first.config_chain == ("cfg/base.yaml", "cfg/child.yaml")
    assert [item.path for item in first.files] == [
        "src/wire.py", "cfg/base.yaml", "cfg/child.yaml"]

    (tmp_path / "cfg" / "base.yaml").write_text(
        "radio:\n  bandwidth: 2\n", encoding="utf-8")
    changed_parent = fingerprint_protocol_implementation(
        tmp_path / "cfg" / "child.yaml", **kwargs)
    assert changed_parent.sha256 != first.sha256

    (tmp_path / "src" / "wire.py").write_text("WIRE=2\n", encoding="utf-8")
    changed_source = fingerprint_protocol_implementation(
        tmp_path / "cfg" / "child.yaml", **kwargs)
    assert changed_source.sha256 != changed_parent.sha256


def test_fingerprint_rejects_cycles_escape_and_duplicate_manifest(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    (tmp_path / "wire.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic"):
        fingerprint_protocol_implementation(
            tmp_path / "a.yaml",
            workspace_root=tmp_path,
            implementation_paths=("wire.py",),
        )
    with pytest.raises(ValueError, match="unique"):
        fingerprint_protocol_implementation(
            tmp_path / "a.yaml",
            workspace_root=tmp_path,
            implementation_paths=("wire.py", "wire.py"),
        )
    outside = tmp_path.parent / "outside-fingerprint.yaml"
    outside.write_text("x: 1\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside workspace"):
            fingerprint_protocol_implementation(
                outside,
                workspace_root=tmp_path,
                implementation_paths=("wire.py",),
            )
    finally:
        outside.unlink()


def test_controller_fingerprint_has_distinct_domain_and_tracks_sources(tmp_path):
    (tmp_path / "controller.py").write_text("VALUE=1\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("value: 1\n", encoding="utf-8")
    kwargs = {
        "workspace_root": tmp_path,
        "implementation_paths": ("controller.py",),
    }
    protocol = fingerprint_protocol_implementation(
        tmp_path / "config.yaml", **kwargs)
    controller = fingerprint_controller_implementation(
        tmp_path / "config.yaml", **kwargs)
    assert controller.sha256 != protocol.sha256
    repeated = fingerprint_controller_implementation(
        tmp_path / "config.yaml", **kwargs)
    assert repeated == controller
    (tmp_path / "controller.py").write_text("VALUE=2\n", encoding="utf-8")
    changed = fingerprint_controller_implementation(
        tmp_path / "config.yaml", **kwargs)
    assert changed.sha256 != controller.sha256
