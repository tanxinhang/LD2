from uav_isac.governance.entrypoint_inventory import build_entrypoint_inventory


def test_entrypoint_inventory_distinguishes_canonical_and_legacy(tmp_path):
    canonical = tmp_path / "uav_isac/interfaces"
    legacy = tmp_path / "tools"
    canonical.mkdir(parents=True)
    legacy.mkdir()
    main_guard = 'if __name__ == "__main__":\n    raise SystemExit(0)\n'
    (canonical / "cli.py").write_text(main_guard, encoding="utf-8")
    (legacy / "old_probe.py").write_text(main_guard, encoding="utf-8")
    (legacy / "library.py").write_text("value = 1\n", encoding="utf-8")

    records = build_entrypoint_inventory(tmp_path)
    by_path = {item.path: item for item in records}

    assert by_path["uav_isac/interfaces/cli.py"].status == "canonical"
    assert by_path["tools/old_probe.py"].status == "legacy_retained"
    assert "tools/library.py" not in by_path

