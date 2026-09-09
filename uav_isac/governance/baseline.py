"""Load immutable behavior characterizations used during V2 migration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_FILE = (
    Path(__file__).resolve().parent / "data" / "characterization_baselines.yaml"
)


@dataclass(frozen=True)
class CharacterizationBaseline:
    name: str
    config: str | None
    seed: int
    frames: int
    semantic_sha256: str


def load_characterization_baselines(
    path: Path | str = DEFAULT_BASELINE_FILE,
) -> Tuple[CharacterizationBaseline, ...]:
    with Path(path).resolve().open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported characterization baseline schema")
    raw_baselines = payload.get("baselines")
    if not isinstance(raw_baselines, dict) or not raw_baselines:
        raise ValueError("baselines must be a non-empty mapping")

    baselines = []
    for name, raw in raw_baselines.items():
        if not isinstance(raw, dict):
            raise ValueError(f"baseline {name!r} must be a mapping")
        seed = raw.get("seed")
        frames = raw.get("frames")
        digest = raw.get("semantic_sha256")
        config = raw.get("config")
        if not isinstance(seed, int) or not isinstance(frames, int) or frames <= 0:
            raise ValueError(f"baseline {name!r} has invalid seed/frames")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"baseline {name!r} has invalid SHA-256")
        if config is not None and not isinstance(config, str):
            raise ValueError(f"baseline {name!r} has invalid config path")
        baselines.append(CharacterizationBaseline(
            name=str(name),
            config=config,
            seed=seed,
            frames=frames,
            semantic_sha256=digest,
        ))
    return tuple(baselines)
