"""Validated registry for active scientific hypotheses and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml


DEFAULT_REGISTRY = Path(__file__).resolve().parent / "data" / "research_programs.yaml"
KNOWN_STATES = {
    "frozen_baseline",
    "shadow_incomplete",
    "candidate",
    "falsified",
    "retired",
}


@dataclass(frozen=True)
class ResearchProgram:
    name: str
    question: str
    state: str
    implementation: str
    profile: str
    evidence: str
    formal_eligible: bool
    next_gate: str


def load_research_programs(
    path: Path | str = DEFAULT_REGISTRY,
) -> Tuple[ResearchProgram, ...]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported research-program registry schema")
    raw_programs = payload.get("programs")
    if not isinstance(raw_programs, list) or not raw_programs:
        raise ValueError("research-program registry must contain programs")

    programs = []
    for raw in raw_programs:
        if not isinstance(raw, dict):
            raise ValueError("each research program must be a mapping")
        missing = [
            name for name in (
                "name", "question", "state", "implementation", "profile",
                "evidence", "formal_eligible", "next_gate",
            )
            if name not in raw
        ]
        if missing:
            raise ValueError(f"research program omits fields: {missing}")
        if raw["state"] not in KNOWN_STATES:
            raise ValueError(f"unknown research program state: {raw['state']!r}")
        if not isinstance(raw["formal_eligible"], bool):
            raise ValueError("formal_eligible must be boolean")
        text_fields = (
            "name", "question", "implementation", "profile", "evidence",
            "next_gate",
        )
        if any(not isinstance(raw[name], str) or not raw[name].strip()
               for name in text_fields):
            raise ValueError("research program text fields must be non-empty")
        if raw["formal_eligible"] and raw["state"] != "frozen_baseline":
            raise ValueError(
                "only a frozen_baseline may be eligible for formal evidence")
        programs.append(ResearchProgram(**{
            field: raw[field] for field in ResearchProgram.__dataclass_fields__
        }))

    names = [program.name for program in programs]
    if len(names) != len(set(names)):
        raise ValueError("research program names must be unique")
    return tuple(programs)
