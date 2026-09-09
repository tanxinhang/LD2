from uav_isac.governance.data_inventory import build_data_inventory


def test_inventory_classifies_without_deleting(tmp_path):
    results = tmp_path / "results"
    docs = tmp_path / "docs"
    results.mkdir()
    docs.mkdir()
    (results / "formal.json").write_text("{}", encoding="utf-8")
    (results / "trial_smoke.log").write_text("smoke", encoding="utf-8")
    (results / "unknown.bin").write_bytes(b"123")
    (results / "_archive").mkdir()
    (results / "_archive/old.json").write_text("{}", encoding="utf-8")
    (docs / "paper.md").write_text(
        "Evidence: results/formal.json\n", encoding="utf-8"
    )

    inventory = build_data_inventory(tmp_path, largest_count=10)
    buckets = {item.lifecycle: item.files for item in inventory.buckets}

    assert inventory.total_files == 4
    assert buckets == {
        "legacy": 1,
        "referenced": 1,
        "scratch_candidate": 1,
        "unclassified": 1,
    }
    assert (results / "trial_smoke.log").exists()

