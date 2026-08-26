from pathlib import Path

import pytest

from tools.run_g2_1a_bridge import SYSTEMS, historical_seeds, preflight


def test_all_bridge_systems_have_exact_historical_seed_banks():
    for name, system in SYSTEMS.items():
        seeds = preflight(name, system)
        assert len(seeds) == 100
        assert len(set(seeds)) == 100


def test_historical_seed_reader_rejects_non_100_bank(tmp_path: Path):
    path = tmp_path / "paired_eval.csv"
    path.write_text("eval_episode_seeds\n\"[1, 2]\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="100 unique"):
        historical_seeds(path)
